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
Implements optimization model for the sketching optimization of the
objective function: ``fobj = 1 - (1/num_skvecs) * Re(Tr(<V@Q,U@Q>))``,
where the rectangular matrix ``Q`` hosts othronormal sketching vectors
in its columns. When ``Q = I`` it is just a full AQC problem.
"""

import logging
from typing import Optional, Callable, Union
import time
import numpy as np
from scipy.stats import truncnorm
import qiskit.algorithms.optimizers as optim
import aqc_research.utils as helper
from aqc_research.circuit_transform import ansatz_to_numpy_fast
import aqc_research.checking as chk
from aqc_research.job_executor import run_jobs
import aqc_research.model_sketching.sk_core as skc
import aqc_research.model_sketching.sk_utils as sku
import aqc_research.optimizer as aqcopt


def _full_aqc(*, maxiter: int, thetas_0: np.ndarray, objv: skc.SketchingObjectiveEx) -> dict:
    """Full AQC optimization by the quasi-Newton optimizer."""
    try:
        optimizer = optim.L_BFGS_B(maxiter=maxiter)
        qiskit_res = optimizer.minimize(fun=objv.objective, x0=thetas_0, jac=objv.gradient)
        result = objv.optim_results
        result["cost"] = qiskit_res.fun  # overwrite by optimizer's output
        result["thetas"] = qiskit_res.x  # overwrite by optimizer's output
        result["exit_status"] = "normal"
    except StopIteration:
        result = objv.optim_results
        result["exit_status"] = "early"
    except TimeoutError:
        result = objv.optim_results
        result["exit_status"] = "timeout"
    return result


def _stochastic_aqc(
    *,
    maxiter: int,
    learn_rate: float,
    thetas_0: np.ndarray,
    objv: skc.SketchingObjectiveEx,
    stop_stagnant: aqcopt.NotImproveStopper,
    logger: logging.Logger,
) -> dict:
    """
    Stochastic optimization with multiple restarts. Note, on every invocation
    of the objective function (i.e. on every iteration) a new set of sketching
    vectors is generated.
    """
    max_learn_rate_corrections = 5  # adjusts learning rate this number of times at most
    ini_thetas = thetas_0.copy()
    while maxiter > 0:
        if logger:
            logger.info(f">>>>> learning rate: {learn_rate}")

        try:
            optimizer = optim.ADAM(maxiter=maxiter, lr=learn_rate)
            qiskit_res = optimizer.minimize(fun=objv.objective, x0=ini_thetas, jac=objv.gradient)
            result = objv.optim_results
            result["cost"] = qiskit_res.fun  # overwrite by optimizer's output
            result["thetas"] = qiskit_res.x  # overwrite by optimizer's output
            result["exit_status"] = "normal"
            break
        except aqcopt.StagnantOptimizationWarning:
            result = objv.optim_results
            max_learn_rate_corrections -= 1
            if max_learn_rate_corrections > 0:  # restart with smaller learning rate
                stop_stagnant.reset()
                learn_rate *= 0.5
                np.copyto(ini_thetas, result["thetas"])
            else:
                stop_stagnant.disable()  # proceed without further restarts
        except StopIteration:
            result = objv.optim_results
            result["exit_status"] = "early"
            break
        except TimeoutError:
            result = objv.optim_results
            result["exit_status"] = "timeout"
            break

        # Every subsequent epoch starts with fewer number of max. iterations.
        maxiter -= objv.num_iterations

    if result.get("exit_status", None) is None:
        result["exit_status"] = "premature"
    return result


def _single_simulation(job_index: int, config: dict) -> dict:
    """
    Runs a single simulation possibly inside a separate (parallel) process.

    Args:
        job_index: unique index of a parallel process.
        config: dictionary of data and parameters required for simulation.

    Returns:
        dictionary of simulation results.
    """
    logger = helper.create_logger("job_0") if job_index == 0 else None
    circ = sku.create_ansatz(
        num_qubits=config["num_qubits"],
        num_layers=config["num_layers"],
        circuit_layout=config["circuit_layout"],
        logger=logger,
    )
    dim = circ.dimension
    maxiter = int(config["maxiter"])
    thetas_0 = np.asarray(truncnorm.rvs(a=-1, b=1, size=circ.num_thetas) * np.pi)
    skvecs = skc.skvecs_generator(
        str(config["skvecs_type"]), int(config["num_skvecs"]), config["su_target"]
    )
    full_aqc = bool(skvecs.num_skvecs == dim)

    stop_stagnant = None if full_aqc else aqcopt.NotImproveStopper(num_iters=40)
    objv = skc.SketchingObjectiveEx(
        circ=circ,
        skvecs=skvecs,
        enable_stats=True,
        grad_scaler=None,  # aqcopt.GradientAmplifier(verbose=verbose),
        stop_timeout=aqcopt.TimeoutStopper(time_limit=config["time_limit"]),
        stop_stagnant=stop_stagnant,
        stop_small_fobj=aqcopt.SmallObjectiveStopper(fobj_thr=1e-2),
        logger=logger,
    )

    if full_aqc:
        result = _full_aqc(maxiter=maxiter, thetas_0=thetas_0, objv=objv)
    else:
        result = _stochastic_aqc(
            maxiter=maxiter,
            learn_rate=float(config["learn_rate"]),
            thetas_0=thetas_0,
            objv=objv,
            stop_stagnant=stop_stagnant,
            logger=logger,
        )

    ansatz_matrix = ansatz_to_numpy_fast(circ, result["thetas"])
    result["fidelity"] = sku.fidelity(ansatz_matrix, config["su_target"])
    result["nit"] = result["num_iters"]
    result["ini_thetas"] = thetas_0
    result["stats"] = objv.statistics
    return result


def aqc_sketching(
    *,
    num_qubits: int,
    num_layers: int,
    num_skvecs: int,
    circ_layout: str,
    maxiter: int,
    learn_rate: float,
    skvecs_type: str,
    target_name_or_func: Union[str, Callable[[int], np.ndarray]],
    result_folder: str,
    parametric_depth: int = int(3),
    seed: int = int(round(time.time())),
    time_limit: int = int(-1),
    num_simulations: int = helper.num_cpus(),
    num_jobs: int = helper.num_cpus(),
    tag: str = "",
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Runs a number of parallel simulations with different initial guesses.

    Args:
        num_qubits: number of qubits.
        num_layers: number of layers of 2-qubit unit-blocks in ansatz.
        num_skvecs: number of sketched vectors in a sample.
        circ_layout: circuit layout; see the function ``supported_layouts()``;
                     currently one of ['spin','line','cyclic_spin','cyclic_line'].
        maxiter: max. number of iterations.
        learn_rate: initial learning rate for gradient descent optimizer,
                    which is used when optimization over a subspace in done (sketching).
        skvecs_type: type of sketching vectors generator, ['full','rand','alt','eigen'].
        target_name_or_func: name of the target matrix from a list of predefined ones or
                             a user supplied function that generates it; prototype:
                             ``def my_target_gen(num_qubits: int) -> np.ndarray``.
        result_folder: parent folder for storing the output results.
        parametric_depth: if a target generated from parametrized ansatz was selected,
                          ``target_name_or_func = 'parametric'``, then this parameter
                          provides the number of *layers* in parametric circuit.
        seed: seed for random generator.
        time_limit: time limit for simulation or -1 (no limit).
        num_simulations: number of independent simulations.
        num_jobs: number of jobs executed simultaneously or -1 (utilizes all CPUs).
        tag: string tag that makes simulation results distinguishable.
        logger: logging object.

    Returns:
        output folder where results have been stored.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(num_layers)
    assert chk.is_int(num_skvecs, num_skvecs > 0)
    assert isinstance(circ_layout, str) and circ_layout in sku.supported_layouts()
    assert chk.is_int(maxiter, maxiter > 0)
    assert chk.is_float(learn_rate, 0 < learn_rate < 1)
    assert isinstance(skvecs_type, str)
    assert callable(target_name_or_func) or isinstance(target_name_or_func, str)
    assert chk.is_str(result_folder, len(result_folder) > 0)
    assert chk.is_int(parametric_depth, parametric_depth >= 1)
    assert chk.is_int(seed)
    assert chk.is_int(time_limit)
    assert chk.is_int(num_simulations, num_simulations >= 1)
    assert chk.is_int(num_jobs, num_jobs == -1 or num_jobs >= 1)
    assert chk.is_str(tag)
    assert chk.none_or_type(logger, logging.Logger)

    # Some common initialization.
    if logger is None:
        logger = helper.create_logger(__file__)

    # Make the output directory by adding sub-folders to the initial (parent) path.
    np.random.seed(seed)
    result_folder = helper.prepare_output_folder(
        result_dir=result_folder,
        num_qubits=num_qubits,
        script_path=__file__,  # this script will be saved along with results
        tag=tag,
    )
    helper.print_options(vars(), logger, numeric_or_str=True)

    # Create a target matrix. User can provide an external target generator.
    target_mat, su_target = sku.create_target_matrix(
        num_qubits=num_qubits,
        target_name_or_func=target_name_or_func,  # user can provide generator
        num_layers=parametric_depth,  # for 'parametric' choice
        circuit_layout=circ_layout,  # for 'parametric' choice
        logger=logger,
    )

    # Run parallel simulations with different initial guesses.
    config = {
        "num_qubits": int(num_qubits),
        "num_layers": int(num_layers),
        "num_skvecs": int(num_skvecs),
        "circuit_layout": circ_layout,
        "maxiter": int(maxiter),
        "learn_rate": float(learn_rate),
        "skvecs_type": str(skvecs_type),
        "time_limit": int(time_limit),
        "su_target": su_target,
    }
    results = run_jobs(
        configs=[config] * num_simulations,
        seed=seed,
        job_function=_single_simulation,
        tolerate_failure=True,
        num_jobs=num_jobs,
    )

    # Pick up the best solution and save all useful artifacts.
    sku.postprocess_and_save_results(
        num_qubits=num_qubits,
        results=results,
        target_mat=target_mat,
        su_target=su_target,
        output_dir=result_folder,
        logger=logger,
    )
    return result_folder
