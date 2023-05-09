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
Generic useful utilities.
"""

import sys
import os
import traceback
import inspect
import logging
import datetime
import shutil
import numbers
from pprint import pprint, pformat
from time import perf_counter
from typing import Optional, Callable, Any, List, Dict, Union
import numpy as np
import pandas as pd
import aqc_research.checking as chk


def num_qubits_from_size(size: int) -> int:
    """Computes the number of qubits from a state size."""
    assert chk.is_int(size, size >= 0)
    num_qubits = int(round(np.log2(float(max(size, 1)))))
    if size != 2**num_qubits:
        raise ValueError("'size' argument is not a power of 2 value")
    return num_qubits


def num_cpus() -> int:
    """Returns the number of CPUs available in the system."""
    ncpus = os.cpu_count()
    if isinstance(ncpus, int):
        return int(ncpus)
    print("WARNING: cannot determine the number of CPUs, defaults to 1")
    return int(1)


def rand_circuit(num_qubits: int, depth: int) -> np.ndarray:
    """
    Generates a random circuit of unit blocks.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(depth, depth >= 0)
    blocks = np.tile(np.arange(num_qubits).reshape(num_qubits, 1), depth)
    for i in range(depth):
        np.random.shuffle(blocks[:, i])
    return blocks[0:2, :].copy()


def rand_thetas(num_thetas: int) -> np.ndarray:
    """
    Creates an array of random parameters of a parametric circuit.
    """
    assert chk.is_int(num_thetas, num_thetas > 0)
    return np.pi * (2 * np.random.rand(num_thetas) - 1)


def rand_state(num_qubits: int) -> np.ndarray:
    """
    Generates a random quantum state.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    dim = 2**num_qubits
    state = np.random.rand(dim) + 1j * np.random.rand(dim)
    state /= np.linalg.norm(state)
    return state


def zero_state(num_qubits: int) -> np.ndarray:
    """
    Generates the quantum state |0>.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    state = np.zeros(2**num_qubits, dtype=np.cfloat)
    state[0] = 1
    return state


def create_logger(module_name: str) -> logging.Logger:
    """
    Creates instance of logging object.

    Args:
        module_name: use "__file__".

    Returns:
        logger instance.
    """
    logger = logging.getLogger(os.path.basename(module_name))
    logger.setLevel(logging.DEBUG)
    logging.addLevelName(logging.INFO, "info")
    logging.addLevelName(logging.WARNING, "Warning")
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger


def print_dot():
    """Prints a dot, useful to show progress of a long computation."""
    print(".", end="", flush=True)


class UserExit:
    """
    User can show intention to terminate gracefully a long-running simulation
    by creating an empty file "aqc_exit" in the current working directory.
    The termination is not necessary immediate. It might take a time before
    the presence of the indicator file is checked.
    """

    def __init__(self, print_banner: bool):
        self._indicator_file = "aqc_exit"
        if os.path.isfile(self._indicator_file):
            os.remove(self._indicator_file)
        if print_banner:
            print(
                f"\n{'*' * 100}\n"
                f"Create an empty file '{self._indicator_file}' for "
                f"early and graceful termination of script execution"
                f"\n{'*' * 100}\n"
            )

    def terminate(self) -> bool:
        """Tries to termination execution if indicator file has been created."""
        if os.path.isfile(self._indicator_file):
            print("!!!!! WARNING: user requested early termination !!!!!")
            return True
        return False


class MyTimer:
    """
    Simple timer which does not clutter the code. Note, execution time is
    accumulated across many runs given the same metric name.
    Example:
        timer = MyTimer()
        with timer(metric_name="job_name1"):
            RunJob1()
        with timer(metric_name="job_name2"):
            RunJob2()
        print(timer.all_metrics())

    Alternatively, one can use a couple of tic()/toc() functions:
        timer = MyTimer()
        timer.tic("job_name1")
        RunJob1()
        timer.toc()
        print(timer.all_metrics())
    """

    def __init__(self, full_time: bool = True):
        """
        Args:
            full_time: reports the overall accumulated time, if non-zero,
                       otherwise the mean time across multiple calls.
        """
        assert isinstance(full_time, bool)
        self._metrics = dict({})
        self._name = None
        self._tic = None
        self._full_time = full_time

    def __call__(self, metric_name: str):
        """Function must be invoked every time one gauges execution time."""
        assert isinstance(metric_name, str) and self._tic is None
        self._name = metric_name
        return self

    def __enter__(self):
        """Invoked by "with operator" at the beginning."""
        self._tic = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Invoked by "with operator" upon completion."""
        if self._name not in self._metrics:
            self._metrics[self._name] = [float(0), int(0)]
        self._metrics[self._name][0] += float(perf_counter() - self._tic)
        self._metrics[self._name][1] += 1
        self._name = None
        self._tic = None
        return exc_type is None

    def tic(self, metric_name: str):
        """Manually starts measuring of a time interval."""
        # pylint: disable=unnecessary-dunder-call
        self.__call__(metric_name)
        self.__enter__()

    def toc(self):
        """Manually ends measuring of a time interval."""
        self.__exit__(None, None, None)

    def all_metrics(self) -> dict:
        """Returns all accumulated performance metrics, if available."""
        return {k: self.metric(k) for k, _ in self._metrics.items()}

    def rounded_metrics(self, decimals: int = 6) -> dict:
        """Returns formatted performance metrics rounded to precision."""
        assert chk.is_int(decimals, decimals >= 0)
        metrics = self.all_metrics()
        if len(metrics) == 0:
            return dict({})
        max_val = max(metrics.values())
        n = int(round(np.floor(np.log10(max(max_val, 1.0)))))
        n += decimals + 2
        return {k: f"{v:{n}.{decimals}f}" for k, v in metrics.items()}

    def metric(self, metric_name: str) -> float:
        """Returns specific metric given its name."""
        m = self._metrics[metric_name]
        return m[0] if self._full_time else m[0] / float(max(m[1], 1))


def temporary_code(message: Optional[str] = None):
    """
    Marker that prints a warning message when temporary code is executed.
    Useful reminder that a certain piece of code must be removed as soon as
    debugging or testing has been done.
    """
    i = inspect.getframeinfo(inspect.currentframe().f_back)
    print("!" * 80)
    print(f"T E M P O R A R Y code at:\n{i.filename} : {i.lineno}")
    if isinstance(message, str):
        print(message)
    print("!" * 80)


def script_entry_point(
    main_func: Callable[..., Union[None, Any]],
    options: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
):
    """
    Implements entry point where execution of every script commences.
    This function invokes ``main_func(options, **kwargs)`` and handles exceptions.
    Example:
    ``
    if __name__ == "__main__":
        script_entry_point(my_main_function, options, logger, optional_arguments)
    ``

    Args:
        main_func: main function of the script.
        options: user supplied options as a class or a namespace.
        logger: logger instance or None.
        kwargs: additional parameters passed to the ``main_func``, the first
                one always stands for user options.
    """
    tic = perf_counter()
    try:
        assert callable(main_func)
        assert logger is None or isinstance(logger, logging.Logger)

        main_func(options, **kwargs)
        print("")
        ok = "finished normally"
        if logger:
            logger.info(ok)
        else:
            print(ok)
    except Exception:
        msg = f"\n{traceback.format_exc()}\n"
        if logger:
            logger.error(msg)
        else:
            print(msg)
    finally:
        print("\n\n")
        tm = f"Total execution time: {perf_counter() - tic:0.2f}"
        if logger:
            logger.info(tm)
        else:
            print(tm)
        print("\n\n\n")


def prepare_output_folder(result_dir: str, num_qubits: int, script_path: str, tag: str) -> str:
    """
    Makes the output directory and returns its path. It also stores the
    script, which implements the numerical experiment being conducted,
    into the output folder for completeness.

    Args:
        result_dir: output folder for results.
        num_qubits: number of qubits.
        script_path: path to the script that implements the numerical experiment.
        tag: additional tag that helps to distinguish the results.

    Returns:
        path to output directory.
    """
    assert isinstance(result_dir, str)
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_str(script_path, os.path.isfile(script_path))
    now = str(datetime.datetime.now().replace(microsecond=0))
    now = now.replace(":", ".").replace(" ", "_")
    output_dir = os.path.join(result_dir, f"{num_qubits}qubits", now)
    if isinstance(tag, str) and len(tag) > 0:
        output_dir = output_dir + "_" + tag
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(script_path, os.path.join(output_dir, os.path.basename(script_path)))
    return output_dir


def print_options(
    opts: dict, logger: Optional[logging.Logger] = None, numeric_or_str: bool = False
):
    """
    Prints out options defined as a dictionary. In case of the dictionary of
    class attributes the internal variables and methods will be filtered out.

    Args:
        opts: dictionary of parameters.
        logger: logging object.
        numeric_or_str: prints numerical or string attributes only, if enabled.
    """

    def _filter_func(_k_, _v_) -> bool:
        return not _k_.startswith("__") and (
            not numeric_or_str or isinstance(_v_, (str, numbers.Number))
        )

    opts = {k: v for k, v in opts.items() if _filter_func(k, v)}
    txt = f"\n{'-' * 80}\nOptions:\n{'-' * 80}\n{pformat(opts)}\n{'-' * 80}\n"
    if isinstance(logger, logging.Logger):
        logger.info(txt)
    else:
        pprint(txt)


def sort_and_print_summary(num_qubits: int, results: List[Dict]) -> List[Dict]:
    """
    Sorts the input results in-place by cost and prints out a brief summary.

    Args:
        num_qubits: number of qubits.
        results: list of results of individual simulations.

    Returns:
        sorted results (the second input argument).
    """
    assert chk.is_int(num_qubits)
    assert chk.is_list(results) and chk.is_dict(results[0])
    results.sort(key=lambda x: x["cost"])
    best_result = results[0]
    assert chk.float_1d(best_result["thetas"])
    assert chk.block_structure(num_qubits, best_result["blocks"])
    pd.set_option("display.max_rows", None)
    summary = pd.DataFrame(results, columns=["cost", "num_iters", "time"])
    print(f"\n{'-' * 24}\nSorted valid results:\n{summary}\n")
    return results


def copy_file_to_folder(directory: str, filename: str):
    """
    Copies existing file to existing directory.

    Args:
        directory: destination directory.
        filename: path to the source file.
    """
    if not os.path.isdir(directory):
        raise IOError("destination directory does not exist")
    if not os.path.isfile(filename):
        raise IOError("source file does not exists")
    shutil.copy(filename, os.path.join(directory, os.path.basename(filename)))


def logi(logger: logging.Logger, message: str):
    """Calls logger.info() without f-string warning about lazy formatting."""
    logger.info(str(message))
