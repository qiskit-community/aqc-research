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
Implements optimization model for coordinate descent optimization of the *full
AQC* objective function: ``fobj = 1 - |<V(thetas),U>|^2 / dim^2``. 
"""

import logging
from typing import Optional, Callable, Union
import time
import numpy as np
from scipy.stats import truncnorm
from aqc_research.circuit_transform import ansatz_to_numpy_fast
import aqc_research.checking as chk
from aqc_research.job_executor import run_jobs
import aqc_research.utils as helper
import aqc_research.model_sketching.sk_utils as sku
import aqc_research.optimizer as aqcopt
from aqc_research.core_op_matrix import coord_descent_single_sweep


def _single_simulation(job_index: int, config: dict) -> dict:
    """
    Runs a single simulation possibly inside a separate (parallel) process.

    Args:
        job_index: unique index of a parallel process.
        config: dictionary of data and parameters required for simulation.

    Returns:
        dictionary of simulation results.
    """
    logger: Union[logging.Logger, None] = None
    if bool(job_index == 0):
        logger = helper.create_logger("job_0")

    thetas_change_threshold = 1e-8  # threshold on change of theta parameters
    enable_stats = True
    target = config["su_target"]
    workspace = np.zeros(3 * target.size, dtype=target.dtype)

    stop_timeout = aqcopt.TimeoutStopper(time_limit=config["time_limit"])
    stop_small_fobj = aqcopt.SmallObjectiveStopper(fobj_thr=1e-2)

    circ = sku.create_ansatz(
        num_qubits=config["num_qubits"],
        num_layers=config["num_layers"],
        circuit_layout=config["circuit_layout"],
        logger=logger,
    )
    thetas_0 = np.asarray(truncnorm.rvs(a=-1, b=1, size=circ.num_thetas) * np.pi)
    thetas, prev_thetas = thetas_0.copy(), thetas_0.copy()
    fobj_best, thetas_best = np.inf, thetas_0.copy()
    nit = int(0)
    fobj_profile = list([])
    result = dict({})

    try:
        # Iterate until convergence or maximum number of iterations reached.
        while nit < config["maxiter"]:
            nit += 1
            np.copyto(prev_thetas, thetas)

            # Update every theta and compute the objective.
            fobj = coord_descent_single_sweep(
                circ=circ,
                thetas=thetas,
                target=target,
                workspace=workspace,
            )
            thetas_change = np.amax(np.abs(thetas - prev_thetas))

            # Memorize the best so far objective function and corresponding thetas.
            if fobj < fobj_best:
                fobj_best = fobj
                np.copyto(thetas_best, thetas)

            # Collect the statistics.
            if enable_stats:
                fobj_profile.append(float(fobj))
            if logger:
                msg = f"iter: {nit:4d}, fobj: {fobj:0.4f}, |dtheta|: {thetas_change:0.5f}"
                logger.info(msg)

            # Check the stop conditions.
            if stop_timeout:
                stop_timeout.check()
            if stop_small_fobj:
                stop_small_fobj.check(fobj=fobj)
            if thetas_change < thetas_change_threshold:
                break

        result["exit_status"] = "normal"
    except StopIteration:
        result["exit_status"] = "early"
    except TimeoutError:
        result["exit_status"] = "timeout"
    finally:
        fobj_profile = np.asarray(fobj_profile, dtype=np.float32)
        fid = sku.fidelity(ansatz_to_numpy_fast(circ, thetas_best), target)
        result["cost"] = float(fobj_best)
        result["nit"] = int(nit)
        result["num_fun_ev"] = int(nit)
        result["num_grad_ev"] = int(nit)
        result["num_iters"] = int(nit)
        result["ini_thetas"] = thetas_0
        result["thetas"] = thetas_best
        result["entangler"] = circ.entangler
        result["blocks"] = circ.blocks
        result["fidelity"] = fid
        result["stats"] = {"convergence_profile": fobj_profile, "nit": nit}
    return result


def aqc_coordinate_descent(
    *,
    num_qubits: int,
    num_layers: int,
    circ_layout: str,
    maxiter: int,
    target_name_or_func: Union[str, Callable[[int], np.ndarray]],
    result_folder: str,
    parametric_depth: int = int(3),
    seed: int = int(round(time.time())),
    time_limit: int = int(0),
    num_simulations: int = helper.num_cpus(),
    num_jobs: int = helper.num_cpus(),
    tag: str = "",
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Runs a number of parallel jobs with different initial guesses
    using coordinate descent optimization of the following *full AQC*
    objective function: ``fobj = 1 - |<V(thetas),U>|^2 / dim^2``.
    The best result will be picked up and reported, along with other ones,
    at the end of the simulation.

    Args:
        num_qubits: number of qubits.
        num_layers: number of layers of 2-qubit unit-blocks in ansatz.
        circ_layout: circuit layout; see the function ``supported_layouts()``.
        maxiter: max. number of iterations.
        target_name_or_func: name of the target matrix from a list of predefined
                             ones or a user supplied function that generates it.
        result_folder: parent folder for storing the output results.
        parametric_depth: if a target generated from parametrized ansatz was selected,
                          ``target_name_or_func = 'parametric'``, then this parameter
                          provides the number of *layers* in parametric circuit.
        seed: seed for random generator.
        time_limit: time limit for simulation or None (no limit).
        num_simulations: number of independent simulations.
        num_jobs: number of jobs executed simultaneously or -1 (utilizes all CPUs).
        tag: string tag that makes simulation results distinguishable.
        logger: logging object.

    Returns:
        output folder where results have been stored.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(num_layers)
    assert isinstance(circ_layout, str) and circ_layout in sku.supported_layouts()
    assert chk.is_int(maxiter, maxiter > 0)
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
        "circuit_layout": circ_layout,
        "maxiter": int(maxiter),
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
