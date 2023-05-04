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
Helper routines for circuit reconstruction from the best result, computation
of accuracy metrics, storing the results into files and parsing the
command-line arguments.
"""

import os
import time
import pickle
import logging
from argparse import ArgumentParser
from typing import Any, List, Dict, Tuple, Optional, Union, Callable
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd  # faster than regular SVD
from qiskit import QuantumCircuit
from aqc_research.target_generator import available_target_matrix_types
from aqc_research.parametric_circuit import ParametricCircuit
import aqc_research.checking as chk
import aqc_research.circuit_transform as ctr
from aqc_research.circuit_structures import create_ansatz_structure, num_blocks_per_layer
import aqc_research.target_generator as targen
import aqc_research.utils as helper


def _approximation_accuracy(
    target: np.ndarray, circ_matrix: np.ndarray, logger: logging.Logger
) -> dict:
    """
    Computes the accuracy metrics to compare approximating circuit vs target.

    **Note**, ``circ_matrix`` will be modified inside this function;
    it should not be used afterwards.

    Args:
        target: target unitary matrix.
        circ_matrix: matrix of approximating circuit (converted from SU to U).
        logger: object for logging.

    Returns:
        dictionary of various accuracy metrics.
    """
    tic = time.perf_counter()
    helper.logi(logger, "computing approximation accuracy ...")

    dim = target.shape[0]
    hsp = np.vdot(circ_matrix, target)  # Hilbert-Schmidt == Tr(V.H @ U)
    hs_cost = 1.0 - np.abs(hsp) / dim
    fidelity_ = (1.0 + np.abs(hsp) ** 2 / dim) / (dim + 1)
    circ_matrix -= target  # V <-- (V - U), not a V anymore !!!
    _, diag, _ = randomized_svd(circ_matrix, n_components=10, random_state=None)
    max_sing = float(np.amax(diag))  # approximate operator norm
    frob = (np.linalg.norm(circ_matrix, "fro") ** 2) / (2 * dim)

    helper.logi(logger, f"done in {time.perf_counter() - tic:0.4f} seconds\n")
    helper.logi(logger, "Approximation accuracy for best result and full matrices:")
    helper.logi(logger, f"HS-cost = 1 - |<V,U>|/dim: {hs_cost:0.8f}")
    helper.logi(logger, f"Fidelity: {fidelity_:0.8f}")
    helper.logi(logger, f"Max. singular value of (V - U): {max_sing:0.8f}")
    helper.logi(logger, f"Largest sing.values of (V - U): {np.round(diag, 3)}")
    helper.logi(logger, f"Frobenius: (|V - U|^2_F)/(2*dim): {frob:0.8f}\n")

    return {
        "hs_cost": hs_cost,
        "fidelity": fidelity_,
        "max_singular": max_sing,
        "frobenius": frob,
    }


def _circuit_from_best_result(
    num_qubits: int,
    best_result: dict,
    target: np.ndarray,
    su_target: np.ndarray,
    logger: logging.Logger,
) -> Tuple[QuantumCircuit, ParametricCircuit, np.ndarray]:
    """
    Creates a quantum circuit, ansatz and its matrix from the best result.
    Also, computes the global phase factor, which is needed to transform the
    matrix of approximating ansatz from SU to U class. Recall, the target
    unitary matrix is in the U class in general.

    Args:
        num_qubits: number of qubits.
        best_result: dictionary of the best result.
        target: target unitary matrix.
        su_target: target matrix possibly converted into SU one.
        logger: object for logging.

    Returns:
        (1) quantum circuit corresponding to the best optimized ansatz (the
        global phase might be non-zero, if the target belongs to U rather
        than SU class); (2) the best optimized ansatz; (3) circuit matrix
        of the best optimized ansatz;
    """
    tic = time.perf_counter()
    helper.logi(logger, "generating the best quantum circuit and its matrix ...")
    circ = ParametricCircuit(
        num_qubits=num_qubits,
        entangler=best_result["entangler"],
        blocks=best_result["blocks"],
    )
    qc = ctr.ansatz_to_qcircuit(circ=circ, thetas=best_result["thetas"])
    circ_matrix = ctr.ansatz_to_numpy_fast(circ=circ, thetas=best_result["thetas"])
    helper.logi(logger, f"done in {time.perf_counter() - tic:0.4f} seconds\n")

    tol = float(np.sqrt(np.finfo(np.float64).eps))
    if not np.allclose(target, su_target, atol=tol, rtol=tol):
        helper.logi(logger, "computing the global phase ...")
        tic = time.perf_counter()
        qc.global_phase = float(np.angle(np.vdot(circ_matrix, target)))
        circ_matrix *= np.exp(1j * qc.global_phase)
        helper.logi(logger, f"done in {time.perf_counter() - tic:0.4f} seconds")
        helper.logi(logger, f"global phase factor (angle): {qc.global_phase:0.6f}\n")

    return qc, circ, circ_matrix


def fidelity(circuit_mat: np.ndarray, target_mat: np.ndarray) -> float:
    """
    Computes fidelity between the matrix of approximating ansatz and the target
    one: ``fidelity = (1 + |Tr(V.H U)|^2 / 2^n) / (2^n + 1)``.
    """
    assert chk.complex_2d_square(circuit_mat) and chk.complex_2d_square(target_mat)
    assert circuit_mat.shape == target_mat.shape

    dim = circuit_mat.shape[0]
    return (1 + np.abs(np.vdot(circuit_mat, target_mat)) ** 2 / dim) / (dim + 1)


def postprocess_and_save_results(
    *,
    num_qubits: int,
    results: List[Dict],
    target_mat: np.ndarray,
    su_target: np.ndarray,
    output_dir: str,
    logger: logging.Logger,
):
    """
    Generates a quantum circuit from the best optimization result, computes
    several accuracy metrics and saves all useful artefacts into the files.

    Args:
        num_qubits: number of qubits.
        results: results obtained from all parallel simulations and
                 sorted by the cost value in ascending order.
        target_mat: target matrix.
        su_target: target matrix converted into the one from SU class; *note*,
                   we actually approximate only SU matrix at the moment.
        output_dir: output directory for all the results.
        logger: object for logging.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert isinstance(results, list) and isinstance(results[0], dict)
    assert chk.complex_2d_square(target_mat)
    assert chk.complex_2d_square(su_target)
    assert isinstance(output_dir, str)
    assert isinstance(logger, logging.Logger)

    # Print out a summary of results.
    results.sort(key=lambda x: x["cost"])
    columns = ["cost", "fidelity", "nit", "time", "exit_status", "status"]
    if results[0].get("fidelity", None) is None:
        columns.pop(1)
    summary = pd.DataFrame(results, columns=columns)
    pd.set_option("display.max_rows", None)
    helper.logi(logger, f"\n{'-' * 24}\nSorted valid results:\n{summary}\n")

    # Build the quantum circuit, ansatz and its matrix from the best result.
    best_result = results[0]
    qc, circ, circ_matrix = _circuit_from_best_result(
        num_qubits=num_qubits,
        best_result=best_result,
        target=target_mat,
        su_target=su_target,
        logger=logger,
    )

    # Compute approximation accuracy. The circuit matrix will be modified!
    acc_metrics = _approximation_accuracy(target=target_mat, circ_matrix=circ_matrix, logger=logger)

    # Save all the results into a file.
    with open(os.path.join(output_dir, "simulation_results.pkl"), "wb") as fld:
        pickle.dump(
            {
                "sorted_results": results,
                "best_result": {
                    "qcircuit": qc,  # quantum circuit equivalent to ansatz
                    "ansatz": circ,  # approximating ansatz
                    "thetas": best_result["thetas"],  # best angular parameters
                    "accuracy_metrics": acc_metrics,  # accuracies
                },
                "target_matrix": target_mat,
            },
            fld,
            protocol=4,  # compatibility with older Pythons
        )

    # Store optimized quantum circuit into a separate file for convenience.
    with open(os.path.join(output_dir, "qcircuit.pkl"), "wb") as fld:
        pickle.dump(qc, fld, protocol=4)

    helper.logi(logger, f"simulation results have been stored in the folder: {output_dir}\n")


def create_ansatz(
    *,
    num_qubits: int,
    num_layers: int,
    circuit_layout: str,
    connectivity: str = "full",
    block_repeat: int = 1,
    entangler: str = "cx",
    logger: Optional[logging.Logger] = None,
) -> ParametricCircuit:
    """
    Creates an instance of parametrized ansatz circuit.

    Default ansatz has a regular, layered structure with the following depth:
    ``circuit depth = (number of layers) * (number of blocks in a layer)``.
    For details see the papers: https://arxiv.org/abs/2106.05649 ,
    https://arxiv.org/abs/2205.04025

    The notion of layer is applicable to a regular circuit layout, e.g. "spin"
    structure. In the latter case, layer is formed by a subset of consecutive
    2-qubit unit-blocks where every pair of adjacent qubits is connected by a
    block. That is, there are ``num_qubits - 1`` unit-blocks in a layer.

    One can also choose "cyclic_spin" layout, additionally connecting the first
    and the last qubits by one extra block, with ``num_qubits`` blocks in a layer.

    Args:
        num_qubits: number of qubits.
        num_layers: number of layers of 2-qubit unit-blocks in ansatz.
        circuit_layout: circuit layout; see the function ``supported_layouts()``.
        connectivity: type of inter-qubit connectivity, {"full", "line"}.
        block_repeat: unit-blocks can repeat 1, 2 or 3 times one after another
                      while being connected to the same couple of qubits.
        entangler: type of entangling gate, one of: ["cx", "cz", "cp"].
        logger: logging object.

    Returns:
        instance of parametric ansatz.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(num_layers)
    if not bool(num_layers >= 1):
        raise ValueError("expects: num_layers >= 1")

    bpl = num_blocks_per_layer(num_qubits, circuit_layout)
    circuit_depth = int(max(1, num_layers)) * bpl

    blocks = create_ansatz_structure(
        num_qubits=num_qubits,
        layout=circuit_layout,
        connectivity=connectivity,
        depth=circuit_depth,
        block_repeat=block_repeat,
        logger=logger,
    )
    circ = ParametricCircuit(num_qubits, entangler=entangler, blocks=blocks)

    if logger:
        helper.logi(
            logger,
            f"ansatz layout: {circuit_layout}, depth = {circ.num_blocks}, "
            f"number of parameters: {circ.num_thetas}",
        )
    return circ


def create_target_matrix(
    *,
    num_qubits: int,
    target_name_or_func: Union[str, Callable[[int], np.ndarray]],
    num_layers: int,
    circuit_layout: str,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a target matrix given either its name (then one of the matrices
    with predefined structure will be created) or user-supplied function,
    which generates a target unitary matrix taking the number of qubits as
    a single argument.

    **Note**, AQC approximates for SU target matrix, not a general unitary one.
    For this reason we generate two matrices: the target one itself and its SU
    version obtained via multiplication by a unit-phase factor such that
    ``det(su_target) = 1``.

    Args:
        num_qubits: number of qubits.
        target_name_or_func: name of the target matrix from a list of predefined
                             ones or a user supplied function that generates it.
        num_layers: number of circuit layers, if 'parametric' target type
                    was selected, ignored otherwise.
        circuit_layout: circuit layout, if 'parametric' target type
                        was selected, ignored otherwise.
        logger: logging object.

    Returns:
        (1) target unitary matrix; (2) SU version of the same target matrix.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(num_layers, num_layers > 0)
    if not (callable(target_name_or_func) or isinstance(target_name_or_func, str)):
        raise ValueError("'target_name_or_func' must be either a function or a string")

    tic = time.perf_counter()
    if callable(target_name_or_func):  # uses a custom target matrix generator
        helper.logi(logger, "target: user-supplied generator")
        target_mat = target_name_or_func(num_qubits)

    elif target_name_or_func == "parametric":  # target from a similar circuit
        helper.logi(logger, f"target: {target_name_or_func}")
        circ = create_ansatz(
            num_qubits=num_qubits,
            num_layers=num_layers,
            circuit_layout=circuit_layout,
            logger=logger,
        )
        target_thetas = np.random.uniform(0, 2 * np.pi, circ.num_thetas)
        target_mat = ctr.ansatz_to_numpy_fast(circ, target_thetas)

    else:  # pick-up target matrix from a small set of predefined ones
        helper.logi(logger, f"target: {target_name_or_func}")
        target_mat = targen.make_target_matrix(target_name_or_func, num_qubits)

    helper.logi(logger, f"target has been created in {time.perf_counter() - tic: 0.3f} secs")

    # Make SU matrix from the general unitary one.
    tic = time.perf_counter()
    su_target = targen.make_su_matrix(target_mat)  # SU target matrix
    helper.logi(logger, f"SU target has been created in {time.perf_counter() - tic: 0.3f} secs")

    return target_mat, su_target


def supported_layouts() -> List[str]:
    """Returns the list of so far supported layouts."""
    return ["spin", "line", "cyclic_spin", "cyclic_line"]


def get_commandline_args(parser: ArgumentParser, logger: logging.Logger) -> Any:
    """Parses and returns the command-line arguments."""
    assert isinstance(parser, ArgumentParser)
    ncpus = helper.num_cpus()
    parser.add_argument(
        "-n",
        "--num_qubits",
        default=int(5),
        type=int,
        help="number of qubits",
        metavar="",
    )
    targ_types = available_target_matrix_types() + ["parametric"]
    parser.add_argument(
        "-t",
        "--target",
        default="parametric",
        type=str,
        help=f"target name, one of: {targ_types}",
        metavar="",
    )
    parser.add_argument(
        "-s",
        "--num_simuls",
        default=ncpus,
        type=int,
        help="total number of simulations with different initial guesses",
        metavar="",
    )
    parser.add_argument(
        "-j",
        "--num_jobs",
        default=ncpus,
        type=int,
        help="number of parallel jobs executed simultaneously",
        metavar="",
    )
    parser.add_argument(
        "-o",
        "--timeout",
        default=-1,
        type=int,
        help="timeout in seconds; non-positive value implies no timeout",
        metavar="",
    )
    parser.add_argument(
        "-g",
        "--tag",
        default="",
        type=str,
        help="tag that makes the name of simulation results distinguishable",
        metavar="",
    )
    cargs = parser.parse_args()
    assert 2 <= cargs.num_qubits <= 16
    assert cargs.target in targ_types
    assert 1 <= cargs.num_simuls <= 100 * ncpus
    assert cargs.num_jobs <= ncpus
    cargs.num_jobs = min(cargs.num_jobs, cargs.num_simuls)
    helper.logi(logger, f"\nCommand-line arguments: {cargs.__dict__}\n\n")
    return cargs
