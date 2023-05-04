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
Utilities for running the parallel simulations.
"""

import traceback
import inspect
import sys
from time import perf_counter
from typing import Callable, List, Dict
from joblib import Parallel, delayed
import numpy as np
import aqc_research.checking as chk


def is_debugging() -> bool:
    """
    Returns:
        non-zero if a script is being executed by debugger.
    """
    for frame in inspect.stack():
        if frame[1].endswith("pdb.py"):
            print(f"\n{'!' * 31}\n!!! Running in debugger ... !!!\n{'!' * 31}\n")
            return True
    return False


def _job_function_wrapper(
    job_index: int, config: Dict, seed: int, job_function: Callable[[int, Dict], Dict]
) -> Dict:
    """
    Runs a single simulation (job), possibly in a separate process.

    Args:
        job_index: unique index of a (parallel) job.
        config: simulation parameters.
        seed: a basic seed for pseudo-random generator.
        job_function: function that implements the simulation model.

    Returns:
        dictionary of results; contains at least three entries:
        1) executions time in seconds ("time", float).
        2) simulation status ("status", str); either "ok ..." or an error message.
        3) job index ("job_index", int).
        4) seed used for pseudo-random generator ("seed", int).
    """
    try:
        assert chk.is_int(job_index, job_index >= 0)
        assert isinstance(config, dict)
        assert chk.is_int(seed)
        assert callable(job_function)

        seed = seed + 7 * (job_index + 1)  # unique seed per process
        np.random.seed(seed)
        # th.manual_seed(seed)  # PyTorch seeds
        # th.cuda.manual_seed(seed)
        # th.cuda.manual_seed_all(seed)

        tic = perf_counter()
        result = job_function(job_index, config)
        toc = perf_counter()

        result.update(
            {
                "time": toc - tic,
                "status": "ok",
                "job_index": job_index,
                "seed": seed,
            }
        )
    except Exception:
        print(f"exception in job={job_index},\n", flush=True)
        result = {
            "time": float(-1),
            "status": traceback.format_exc(),
            "job_index": job_index,
            "seed": seed,
        }
    finally:
        sys.stderr.flush()
        sys.stdout.flush()
    return result


def run_jobs(
    configs: List[Dict],
    seed: int,
    job_function: Callable[[int, Dict], Dict],
    *,
    tolerate_failure: bool = False,
    num_jobs: int = -1,
) -> List[Dict]:
    """
    Runs a number of simulations, for every configuration, possibly in parallel.

    Args:
        configs: a list of configurations, where configuration is defined as
                 a dictionary of parameters for every individual simulation.
        seed: a basic seed for pseudo-random generator; inside each parallel
              process it will be used to create a unique one.
        job_function: function that executes an individual simulation; it takes
                      a job index and a dictionary of configuration parameters
                      as an input, and returns a dictionary of simulation results.
        tolerate_failure: tolerate the situation where some (but not all)
                          simulations had failed; if True, only valid simulation
                          results will be returned.
        num_jobs: number of jobs executed simultaneously; should be either -1,
                  for utilization of all CPUs, or a positive integer.

    Returns:
        list of all **valid** simulation results; every individual result
        contains whatever simulation returns plus the following entries:
        1) executions time ("time", float).
        2) simulation status ("status", str) initialized to "ok" state
           for normal accomplishment or to error message on failure.
        3) job index ("job_index", int).
        4) seed that has been used in a simulation ("seed", int).
    """
    assert chk.is_list(configs, len(configs) > 0) and chk.is_dict(configs[0])
    assert callable(job_function)
    assert chk.is_int(num_jobs)
    assert num_jobs == -1 or num_jobs >= 1

    num_jobs = 1 if is_debugging() else num_jobs
    if num_jobs == 1:
        results = []
        for i, c in enumerate(configs):
            results.append(_job_function_wrapper(i, c, seed, job_function))
    else:
        results = Parallel(n_jobs=num_jobs, prefer="processes")(
            delayed(_job_function_wrapper)(i, c, seed, job_function) for i, c in enumerate(configs)
        )

    print("")
    sys.stderr.flush()
    sys.stdout.flush()

    for r in results:
        if not bool(r["status"].startswith("ok")):
            idx, status = r["job_index"], r["status"]
            print(f"Simulation {idx} failed:\n\n{status}\n{'-' * 80}\n\n")

    if sum(map(lambda x: bool(x["status"].startswith("ok")), results)) == 0:
        raise RuntimeError("there is no valid simulation results")

    # Choose a subset of successful simulations, if allowed.
    if tolerate_failure:
        results = [r for r in results if bool(r["status"].startswith("ok"))]

    return results
