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
Utility functions useful for running numerical experiments.
"""

import os
import pickle
import datetime
from pprint import pprint
from typing import Any, List, Dict, Optional, Callable, Union
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
import aqc_research.utils as helper
import aqc_research.mps_operations as mpsop
import aqc_research.checking as chk
from aqc_research.parametric_circuit import ParametricCircuit, TrotterAnsatz
import aqc_research.circuit_transform as ctr
from aqc_research.core_operations import v_mul_vec
from aqc_research.model_sp_lhs.user_options import UserOptions

_logger = helper.create_logger(__file__)


def load_results_from_archive(filename: str) -> List[Dict]:
    """
    Reads simulation results from the pickle archive file.

    **Note**, the data size can be huge (dozens of gigabytes) for a large
    number of qubits.

    Args:
        filename: path to the file of results.
    """
    with open(filename, "rb") as fld:
        data = pickle.load(fld)
    assert isinstance(data, list), "expects archive with a list of results"
    horizons = [r["evol_time1"] for r in data]
    print("Number of time horizons:", len(horizons))
    pprint(f"Time horizons: {horizons}")
    return data


def qcircuit_from_result(result: dict, tol: Optional[float] = 0.0) -> QuantumCircuit:
    """
    Generates the solution quantum circuit from the optimization result.

    Args:
        result: optimization result.
        tol: tolerance level to identify nearly-zero angles; a gate will be
             discarded if having angular parameter close to zero.

    Returns:
        quantum circuit obtained from the ansatz of optimization result.
    """
    assert isinstance(result, dict)
    assert chk.is_float(tol, 0 <= tol < 1)
    assert result["entangler"] == "cx"
    circ = TrotterAnsatz(
        num_qubits=result["num_qubits"],
        blocks=result["blocks"],
        second_order=result["second_order_trotter"],
    )
    return ctr.ansatz_to_qcircuit(circ, result["thetas"], tol=tol)


def get_solution_from_optim_result(
    opts: UserOptions,
    result: dict,
    trotterized: bool,
    state_prep_func: Optional[Callable[[int], QuantumCircuit]] = None,
    trunc_thr: Optional[float] = None,
) -> Union[mpsop.QiskitMPS, np.ndarray]:
    """
    Generates the solution state from the optimization result.

    Args:
        opts: user supplied options defined as a class or namespace.
        result: optimization result.
        trotterized: non-zero if ansatz layout is similar to
                     the Trotter circuit.
        state_prep_func: function returns a quantum circuit that produces
                         an initial state from ``|0>``.
        trunc_thr: optional truncation threshold used to reconstruct MPS state.

    Returns:
        solution state in MPS or classic format.
    """
    assert state_prep_func is None or callable(state_prep_func)
    num_qubits = result["num_qubits"]
    if trotterized:
        circ = TrotterAnsatz(num_qubits, result["blocks"], second_order=opts.second_order_trotter)
    else:
        circ = ParametricCircuit(num_qubits, result["entangler"], result["blocks"])

    # Compute the output state applying the ansatz circuit to the initial state.
    if opts.use_mps:
        qc = ctr.ansatz_to_qcircuit(circ, result["thetas"])
        if state_prep_func is not None:
            qc = state_prep_func(num_qubits).compose(qc)
        if trunc_thr is None:
            trunc_thr = opts.trunc_thr
        assert chk.is_float(trunc_thr, 0 < trunc_thr < 0.1)
        return mpsop.mps_from_circuit(qc, trunc_thr=trunc_thr)
    else:
        if state_prep_func is not None:
            state = ctr.qcircuit_to_state(state_prep_func(num_qubits))
        else:
            state = helper.zero_state(num_qubits)

        workspace = np.ndarray((2, circ.dimension), dtype=np.cfloat)
        state = v_mul_vec(circ, result["thetas"], state, state, workspace)
        return state


def save_optim_results(
    output_dir: str,
    results: List[Dict],
    target: Optional[Union[mpsop.QiskitMPS, np.ndarray]] = None,
    tag: Optional[str] = "",
):
    """
    Saves sorted optimization results into a file. It is assumed that the
    first result is the best one in terms of cost value, i.e. results are sorted.

    Args:
        output_dir: output directory.
        results: all simulation results sorted by cost.
        target: target state in MPS or classical representation.
        tag: file tag that makes it distinguishable.
    """
    assert chk.is_str(output_dir)
    assert all(results[0]["cost"] <= r["cost"] for r in results)

    tag = "" if len(tag) == 0 else ("_" + tag)
    best_cost = f"{results[0]['cost']:0.8f}"
    filename = f"trotter{tag}_n{results[0]['num_qubits']}__c{best_cost}.pkl"
    with open(os.path.join(output_dir, filename), "wb") as fld:
        pickle.dump({"results": results, "target": target}, fld)
        _logger.info("results have been written in the file: %s", fld.name)


def get_commandline_args(parser: ArgumentParser) -> Any:
    """Parses and returns the command-line arguments."""
    assert isinstance(parser, ArgumentParser)
    parser.add_argument(
        "-n",
        "--num_qubits",
        default=5,
        type=int,
        help="number of qubits",
        metavar="",
    )
    parser.add_argument(
        "-t",
        "--target_only",
        action="store_true",
        help="flag: compute target states and exit",
    )
    parser.add_argument(
        "-g",
        "--tag",
        default="",
        type=str,
        help="tag that makes simulation results distinguishable",
        metavar="",
    )
    parser.add_argument(
        "-f",
        "--targets_file",
        default="",
        type=str,
        help="path to a file with precomputed targets",
        metavar="",
    )
    params = parser.parse_args()
    assert 2 <= params.num_qubits
    _logger.info("\nCommand-line arguments: %s\n\n", params.__dict__)
    return params


def prepare_output_folder(opts: UserOptions, script_path: str) -> str:
    """
    Makes the output directory and returns its path. It also stores the
    script, which implements the numerical experiment being conducted,
    into the output folder for completeness as well as user options.

    Args:
        opts: user supplied options defined as a class or namespace.
        script_path: path to the script that implements the numerical experiment.

    Returns:
        path to output directory.
    """
    assert chk.is_int(opts.num_qubits, opts.num_qubits >= 2)
    assert chk.is_str(script_path, os.path.isfile(script_path))
    now = str(datetime.datetime.now().replace(microsecond=0))
    now = now.replace(":", ".").replace(" ", "_")
    output_dir = os.path.join(opts.result_dir, f"{opts.num_qubits}qubits", now)
    if isinstance(opts.tag, str) and len(opts.tag) > 0:
        output_dir = output_dir + "_" + opts.tag
    os.makedirs(output_dir, exist_ok=True)
    helper.copy_file_to_folder(output_dir, script_path)
    with open(os.path.join(output_dir, "user_options.pkl"), "wb") as fld:
        pickle.dump(opts, fld)  # stores user options
    return output_dir


def verify_and_print_summary(num_qubits: int, results: List[Dict]):
    """
    Verifies all optimization results and prints out a brief summary.

    Args:
        num_qubits: number of qubits.
        results: list of results of individual simulations.
    """
    assert chk.is_int(num_qubits)
    assert chk.is_list(results) and chk.is_dict(results[0])

    n = len(results)
    if not all(results[i]["cost"] <= results[i + 1]["cost"] for i in range(n - 1)):
        raise ValueError("simulation results are not sorted by 'cost'")

    best_result = results[0]
    assert chk.float_1d(best_result["thetas"])
    assert chk.block_structure(num_qubits, best_result["blocks"])
    summary = pd.DataFrame(results, columns=["cost", "fidelity", "num_iters", "time"])
    _logger.info("\n%s\nSorted valid results:\n%s\n", str("-" * 24), str(summary.to_string()))


def print_results(results: List[Dict], result_no: Optional[int] = None):
    """
    Prints out either all simulation results or selected one.
    Function is useful for inspecting relatively small archive files.

    Args:
        results: list of simulation results for all time horizons.
        result_no: index of selected result or None (all results).
    """
    if result_no is not None:
        assert chk.is_int(result_no)
        if not bool(0 <= result_no < len(results)):
            raise IndexError("'result_no' is out of range")

    for idx, res in enumerate(results):
        if result_no is None or result_no == idx:
            print(f"\n{'&' * 80}\nHorizon no. {idx}\n{'&' * 80}\n")
            pprint(res)
