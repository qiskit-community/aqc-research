# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Simulation script for Hamiltonian time evolution that combines Trotter
and ASP steps. Namely, we move forward in time by relatively small step
using Trotterization, then approximate the Trotter state by means of ASP
with more shallow circuit than the one used in Trotterization.
In this script:
1) ASP is based on MPS or classic representation of state vectors.
2) Objective function is implemented as a surrogate approximation model that
relies on the single maximal projection onto the subspace of flipped states.
3) Ansatz circuit has the same layout as the Trotter one and is initialized at
the beginning of optimization in such way that every layer in ansatz is similar
to a single Trotter step ('perfect' initialization).
4) We optimize ansatz to make it only slightly better than a Trotter circuit
in order to save computation time.
"""

import os
import pickle
import time
from typing import Tuple, Union, Optional
from pprint import pformat
import numpy as np
import aqc_research.utils as helper
import aqc_research.checking as chk
from aqc_research.model_sp_lhs.objective_base import SpLHSObjectiveBase
from aqc_research.model_sp_lhs.objective_lhs_sur_max import SpSurrogateObjectiveMax
from aqc_research.model_sp_lhs.objective_lhs_sur_fast_mps_trotter import (
    SpSurrogateObjectiveFastMpsTrotter,
)
import aqc_research.optimizer as optim
from aqc_research.circuit_structures import make_trotter_like_circuit
from aqc_research.parametric_circuit import (
    TrotterAnsatz,
    layer_to_block_range,
    first_layer_included,
)
import aqc_research.model_sp_lhs.trotter.trotter as trotop
from aqc_research.model_sp_lhs.trotter.target_states import (
    TargetClassicState,
    TargetMpsState,
    get_target_states,
)
import aqc_research.model_sp_lhs.trotter.trotter_evol_utils as trot_utils
from aqc_research.model_sp_lhs.trotter.trotter_plots import plot_fidelity_profiles
from aqc_research.mps_operations import QiskitMPS, no_truncation_threshold
from aqc_research.model_sp_lhs.trotter.trotter import fidelity
from aqc_research.model_sp_lhs.user_options import UserOptions

_logger = helper.create_logger(__file__)


def _create_objective(
    *,
    opts: UserOptions,
    circ: TrotterAnsatz,
    target: Union[QiskitMPS, np.ndarray],
    layer_range: Union[Tuple[int, int], None],
) -> SpLHSObjectiveBase:
    """Creates an instance of objective function."""
    params = {
        "job_index": int(0),
        "num_qubits": circ.num_qubits,
        "max_flips": int(1),
        "maxiter": opts.maxiter,
        "verbose": opts.verbose,
        "enable_optim_stats": True,
        "num_simulations": int(1),
        "trunc_thr": opts.trunc_thr,
        "state_prep_func": opts.ini_state_func[0],
    }

    # Enable gradient amplification in case of barren plateau. One can also
    # set grad_scaler = None to disable this feature, which can potentially
    # be harmful (but not necessary is) if gradient descent optimizer is used.
    grad_scaler = None
    if opts.enable_grad_scaling:
        grad_scaler = optim.GradientAmplifier(history=5, strong=False, verbose=opts.verbose)

    # Optimization of the first layer of 2-qubit blocks (layer_range[0] == 0)
    # implies optimization of the front layer of 1-qubit gates as well.
    if opts.objective == "sur_max":
        objv = SpSurrogateObjectiveMax(
            user_parameters=params,
            circ=circ,
            block_range=layer_to_block_range(circ, layer_range),
            front_layer=first_layer_included(circ, layer_range),
            verbose=opts.verbose,
            grad_scaler=grad_scaler,
        )
    elif opts.objective == "sur_fast_mps_trotter":
        objv = SpSurrogateObjectiveFastMpsTrotter(
            user_parameters=params,
            circ=circ,
            layer_range=layer_range,
            alt_layers=False,  # keep it False for now
            verbose=opts.verbose,
            grad_scaler=grad_scaler,
        )
    else:
        raise ValueError(f"unknown objective function: {opts.objective}")

    objv.set_target(target)
    return objv


def _calc_fidelity_threshold(
    target: Union[TargetClassicState, TargetMpsState],
    fidelity_thr: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculates fidelity threshold. Sets fidelity threshold to a bit above
    fidelity of the reference state but not too high.

    Args:
        target: target states (ground-truth and reference ones).
        fidelity_thr: desired least fidelity; None implies automatic selection.

    Returns:
        fidelity threshold, and fidelity between |t1> and ground-truth state.
    """
    fid_t1_vs_gt = fidelity(target.t1, target.t1_gt)
    if fidelity_thr is not None:
        assert chk.is_float(fidelity_thr, 0 < fidelity_thr <= 1)
        fid_thr = max(fid_t1_vs_gt, fidelity_thr)
    else:
        fid_thr = 1.03 * fid_t1_vs_gt
    _logger.info("Fidelity threshold: %0.4f", fid_thr)
    return fid_thr, fid_t1_vs_gt


def _model_function(
    *,
    opts: UserOptions,
    num_layers: int,
    evol_time: float,
    target: Union[QiskitMPS, np.ndarray],
    fid_thr: float,
) -> dict:
    """
    Implements approximation model for optimizing a parametric circuit given
    an initial guess and a target state, possibly defined as MPS.
    **Note**, here we use so called 'perfect' initial guess ('thetas_0'),
    which is already sufficiently close to the optimal solution.

    Args:
        opts: user supplied options.
        num_layers: number of layers in the Trotter circuit.
        evol_time: evolution time of the target state.
        target: target state to be approximated.
        fid_thr: fidelity threshold.

    Returns:
        a dictionary of optimization result.
    """
    tic = time.perf_counter()
    assert num_layers >= 1
    assert 0 < fid_thr <= 1
    _logger.info("#layers: %d, evol.time: %0.3f", num_layers, evol_time)

    layer_range = (0, num_layers)  # entire circuit
    blocks = make_trotter_like_circuit(
        num_qubits=opts.num_qubits,
        num_layers=num_layers,
        connectivity="full",
        verbose=bool(opts.verbose),
    )
    circ = TrotterAnsatz(
        num_qubits=opts.num_qubits,
        blocks=blocks,
        second_order=opts.second_order_trotter,
    )
    thetas_0 = trotop.init_ansatz_to_trotter(
        circ=circ,
        thetas=np.zeros(circ.num_thetas),
        evol_time=evol_time,
        delta=opts.delta,
        layer_range=layer_range,
    )
    objv = _create_objective(
        opts=opts,
        circ=circ,
        target=target,
        layer_range=layer_range,
    )
    optimizer = optim.AqcOptimizer(
        optimizer_name="lbfgs",
        maxiter=int(opts.maxiter),
        verbose=opts.verbose,
    )
    result = optimizer.optimize(
        objv=objv,
        circ=circ,
        thetas_0=thetas_0,
        stopper=optim.EarlyStopper(fidelity_thr=fid_thr),
        timeout=optim.TimeoutChecker(time_limit=opts.time_limit),
    )
    result.update(  # add helpful information
        {
            "num_qubits": circ.num_qubits,
            "num_layers": num_layers,
            "entangler": circ.entangler,  # Trotter implies "cx", always
            "time": time.perf_counter() - tic,
        }
    )
    _logger.info("Final objective function value: %0.6f", float(result["cost"]))
    return result


def _time_evolution(
    *,
    opts: UserOptions,
    num_layers: int,
    num_expansions: int,
    target: Union[TargetClassicState, TargetMpsState],
    output_dir: str,
) -> dict:
    """
    The core function that runs the computational model, possibly with
    several expansions of the ansatz circuit. The computation is done for
    chosen time horizon ('target.evol_time') from scratch, that is, here
    we do not re-use any information obtained for the previous time horizon.

    Args:
        opts: user supplied options.
        num_layers: initial number of layers in ansatz circuit (can be increased).
        num_expansions: maximal number of circuit expansions where we add more
                        layers to ansatz to reach better accuracy.
        target: target states (ground-truth and reference ones).
        output_dir: directory for optimization results.

    Returns:
        dictionary of simulation results.
    """
    assert chk.is_int(num_layers, num_layers >= 1)
    assert chk.is_int(num_expansions, num_expansions >= 0)

    _logger.info("\n%s\nEvolution time: %f\n%s", "&" * 60, target.evol_time, "&" * 60)
    assert target.num_trot_steps == opts.trotter_steps[target.my_id]

    fidelity_thr, fid_t1_vs_gt = _calc_fidelity_threshold(
        target=target, fidelity_thr=opts.fidelity_thr
    )

    # Make several expansions of ansatz circuit until either fidelity is high
    # enough or the maximum number of attempts has been done.
    attempt = 0
    while True:
        _logger.info("\n%s\nNumber of layers: %d\n%s", "=" * 40, num_layers, "=" * 40)

        # Obtain |a1> by ansatz optimization.
        tic = time.perf_counter()
        a_state_result = _model_function(
            opts=opts,
            num_layers=num_layers,
            evol_time=target.evol_time,
            target=target.t1_gt,  # ground-truth
            fid_thr=fidelity_thr,
        )
        _logger.info("done |a1> state in %0.3f secs", time.perf_counter() - tic)
        trot_utils.verify_and_print_summary(opts.num_qubits, [a_state_result])

        # Store the current (intermediate) optimization result.
        if opts.save_intermediate_results:
            tag = f"t1_{target.evol_time:0.3f}__nl{num_layers}"
            trot_utils.save_optim_results(output_dir, [a_state_result], target.t1_gt, tag)

        # Compute approximating states.
        a1 = trot_utils.get_solution_from_optim_result(
            opts=opts,
            result=a_state_result,
            trotterized=True,
            state_prep_func=opts.ini_state_func[0],
        )

        # Fidelity computed directly might be slightly lower than that computed
        # in objective function (due to limited precision, especially in MPS).
        fid_a1_vs_gt = trotop.fidelity(a1, target.t1_gt)
        if max(fid_a1_vs_gt, a_state_result.get("fidelity", float(0))) > fidelity_thr:
            break  # fidelity is high enough, proceed to the next time horizon
        if attempt >= num_expansions:
            break  # no more expansions are allowed for the current time horizon

        # Try to improve fidelity by circuit expansion.
        attempt += 1  # one more attempt
        num_layers += 1  # expand ansatz circuit by one layer
        _logger.info("inserting extra unit-block")

    # The final result is recomputed without truncation (i.e. max. precision).
    if opts.use_mps:
        _logger.info("the final result will be recomputed without truncation ...")
        a1 = trot_utils.get_solution_from_optim_result(
            opts=opts,
            result=a_state_result,
            trotterized=True,
            state_prep_func=opts.ini_state_func[0],
            trunc_thr=no_truncation_threshold(),
        )
        fid_a1_vs_gt = trotop.fidelity(a1, target.t1_gt)

    # Make a dictionary of computation results of the current horizon.
    assert num_layers == a_state_result["num_layers"]
    res = dict({})
    res["fid_a1_vs_gt"] = fid_a1_vs_gt  # |a1> vs |t1_gt>
    res["fid_t1_vs_gt"] = fid_t1_vs_gt  # |t1> vs |t1_gt>
    res["fid_a1_vs_t1"] = trotop.fidelity(a1, target.t1)  # |a1> vs |t1>
    res["num_qubits"] = opts.num_qubits
    res["num_layers"] = num_layers
    res["block_reps"] = 3  # always 3 for Trotterized ansatz
    res["entangler"] = str(a_state_result["entangler"])  # always "cx" for Trotter
    res["num_trotter_steps"] = target.num_trot_steps
    res["evol_time1"] = target.evol_time
    res["thetas"] = a_state_result["thetas"].copy()
    res["blocks"] = a_state_result["blocks"].copy()
    res["use_mps"] = bool(opts.use_mps)
    res["second_order_trotter"] = bool(opts.second_order_trotter)
    res["ini_state_func"] = opts.ini_state_func[0]
    res["stats"] = a_state_result.get("stats", None)

    # Print out fidelity measures.
    fids = pformat({k: f"{v:0.6f}" for k, v in res.items() if k.startswith("fid_")})
    _logger.info("\n%s\n%s", fids, str("-" * 80))
    return res


def run_simulation(opts: UserOptions) -> str:
    """
    Runs an experiment that comprises several independent simulations
    for different time horizons.

    Args:
        opts: user options.

    Returns:
        output folder where results have been stored.
    """
    # Initialize the framework.
    helper.print_options(opts.__dict__, _logger)
    output_dir = trot_utils.prepare_output_folder(opts, __file__)
    targets = get_target_states(opts)  # compute or load targets
    if opts.target_only:
        return output_dir  # compute target states and exit

    # Number of precomputed targets can differ from that specified in options.
    targets = targets[0 : min(len(targets), len(opts.trotter_steps))]

    # Object enables graceful early termination upon user request.
    user_exit = helper.UserExit(True)

    # Run simulations for all time horizons separately. Recall, every target
    # has been computed for a particular time horizon.
    all_results = list([])
    for idx, targ in enumerate(targets):
        if user_exit.terminate():
            break  # user requested termination

        # Either user set manually the number of layers per time horizon or
        # the same number of layers will be added on every (big) time step.
        if chk.is_list(opts.manual_num_layers) and len(opts.manual_num_layers) > idx:
            num_layers = int(opts.manual_num_layers[idx])
        else:
            num_layers = int(opts.num_layers_inc * (idx + 1))

        res = _time_evolution(
            opts=opts,
            num_layers=num_layers,
            num_expansions=0,
            target=targ,
            output_dir=output_dir,
        )
        all_results.append(res)

    # Store the simulation results.
    with open(os.path.join(output_dir, "all_results.pkl"), "wb") as fld:
        pickle.dump(all_results, fld)

    # Plot and save the picture of time evolution accuracy vs time horizon.
    plot_fidelity_profiles(
        results=all_results,
        output_dir=output_dir,
        no_print_block_rep=True,
    )
    _logger.info("The output folder: %s", output_dir)
    return output_dir
