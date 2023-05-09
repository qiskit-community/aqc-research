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
Generators of target states and unitary matrices.
"""

from time import perf_counter
from typing import List
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import expm as matrix_exp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import QFT
from aqc_research.parametric_circuit import ParametricCircuit
import aqc_research.utils as utl
import aqc_research.core_operations as cop
import aqc_research.checking as chk

_logger = utl.create_logger(__file__)

# -----------------------------------------------------------------------------
# Target state generators.
# -----------------------------------------------------------------------------


def available_target_state_types() -> List[str]:
    """
    Returns:
        a list of supported target state types.
    """
    return ["parametric", "bare", "random"]


def make_target_state(target_name: str, num_qubits: int) -> np.ndarray:
    """
    Generates and returns a target state vector.

    Args:
        target_name: string that defines the type of target to be generated.
        num_qubits: number of qubits.

    Returns:
        generated target state.
    """
    tic = perf_counter()
    msg = "generates target state from"

    if target_name == "parametric":
        _logger.info("%s random ansatz with random angular parameters", msg)
        circ = ParametricCircuit(
            num_qubits=num_qubits,
            entangler="cx",
            blocks=utl.rand_circuit(
                num_qubits, np.random.randint(2 * num_qubits, 4 * num_qubits + 1)
            ),
        )
        thetas = utl.rand_thetas(circ.num_thetas)  # random
        target = target_state_from_circuit(circ, thetas)

    elif target_name == "bare":
        _logger.info("%s random circuit with CNOT gates only", msg)
        circ = ParametricCircuit(
            num_qubits=num_qubits,
            entangler="cx",
            blocks=utl.rand_circuit(
                num_qubits, np.random.randint(2 * num_qubits, 4 * num_qubits + 1)
            ),
        )
        thetas = np.zeros(circ.num_thetas)  # zeros, i.e. no 1-qubit rotations
        target = target_state_from_circuit(circ, thetas)

    elif target_name == "random":
        _logger.info("%s a random vector", msg)
        target = utl.rand_state(num_qubits)
        target /= np.linalg.norm(target)  # normalize

    else:
        raise ValueError(
            f"unsupported target type, expects one of: "
            f"{available_target_state_types()}, got {target_name}",
        )

    toc = perf_counter()
    _logger.info("target state has been prepared in %0.2f secs", toc - tic)
    return target


def target_state_from_circuit(circ: ParametricCircuit, thetas: np.ndarray) -> np.ndarray:
    """
    Creates a target state by applying parametric circuit to vector |0>.

    Args:
        circ: instance of parametric ansatz.
        thetas: angular parameters of the parametric circuit.

    Returns:
        generated target.
    """
    ini_state = utl.zero_state(circ.num_qubits)  # |0>
    target = np.zeros_like(ini_state)
    workspace = np.zeros((2, circ.dimension), dtype=np.cfloat)
    cop.v_mul_vec(circ=circ, thetas=thetas, vec=ini_state, out=target, workspace=workspace)

    # Should be normalized.
    norm_target = np.linalg.norm(target)
    tol = 3 * float(np.sqrt(np.finfo(np.float64).eps))
    assert np.isclose(norm_target, 1, rtol=tol, atol=tol)

    # Report the overlap between initial (|0>) and target states.
    numer = np.abs(np.vdot(ini_state, target))
    denom = np.linalg.norm(ini_state) * norm_target
    overlap = numer / max(denom, np.finfo(float).eps)
    _logger.info("initial vs target vector overlap: %0.3f", overlap)
    if overlap > 0.9:
        _logger.warning("target state is too close to |0>")

    return target


# -----------------------------------------------------------------------------
# Target unitary matrix generators.
# -----------------------------------------------------------------------------


def available_target_matrix_types() -> List[str]:
    """
    Returns:
        a list of supported target types.
    """
    return [
        "random",
        "random_ps2",
        "random_ps4",
        "random_ps8",
        "random_ps16",
        "random_rank2",
        "random_rank4",
        "random_rank8",
        "random_rank16",
        "mcx",
        "qft",
        "shift1",
        "shift2",
        "shift_half",
        "random_perm",
    ]


def make_target_matrix(target_name: str, num_qubits: int) -> np.ndarray:
    """
    Generates and returns a target unitary matrix.

    Args:
        target_name: string that defines the type of target to be generated.
        num_qubits: number of qubits.

    Returns:
        generated target matrix.
    """
    tic = perf_counter()
    msg = "generates target unitary from"
    dim = 2**num_qubits

    if target_name == "random":
        _logger.info("%s a random matrix", msg)
        target = unitary_group.rvs(dim)

    elif target_name.startswith("random_rank"):
        rank = int("".join(filter(str.isdigit, target_name)))
        assert 0 < rank < dim
        _logger.info("%s a random, rank-%d matrix", msg, rank)
        q_mat = np.random.rand(dim, rank) + 1j * np.random.rand(dim, rank)
        q_mat, _ = np.linalg.qr(q_mat)
        target = matrix_exp(-0.25j * (q_mat @ np.conj(q_mat.T)))

    elif target_name.startswith("random_ps"):
        nps = int("".join(filter(str.isdigit, target_name)))
        assert 0 < nps < dim  # nps - the number of Pauli strings
        _logger.info("%s %d random Pauli strings", msg, nps)
        pms = np.asarray(  # Pauli matrices
            [
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[0, -1j], [1j, 0]],
                [[1, 0], [0, -1]],
            ],
        )
        target = np.zeros((dim, dim), np.cfloat)
        for _ in range(nps):
            pstr = 1
            for __ in range(num_qubits):
                pstr = np.kron(pstr, pms[np.random.randint(0, 4)])
            pstr *= 0.75 * (1 + np.random.rand())  # scale: 0.75 .. 1.5
            target += pstr
        target = matrix_exp(-0.25j * target)

    elif target_name == "mcx":
        _logger.info("%s a single multi-control CNOT gate", msg)
        target = np.eye(dim, dtype=np.cfloat)
        half, last = dim // 2 - 1, dim - 1
        target[half, half], target[half, last] = 0, 1
        target[last, half], target[last, last] = 1, 0

        # Check a small matrix against Qiskit one.
        if num_qubits <= 7:
            qc = QuantumCircuit(num_qubits)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            tol = 100 * np.finfo(np.float64).eps
            assert np.allclose(target, Operator(qc).data, atol=tol, rtol=tol)

    elif target_name == "qft":
        _logger.info("%s a QFT circuit", msg)
        target = Operator(QFT(num_qubits)).data

    elif target_name == "shift1":
        _logger.info(
            "%s a unit matrix with diagonal cyclically shifted one position to the right",
            msg,
        )
        target = np.roll(np.eye(dim, dtype=np.cfloat), 1, axis=1)

    elif target_name == "shift2":
        _logger.info(
            "%s a unit matrix with diagonal cyclically shifted two positions to the right",
            msg,
        )
        target = np.roll(np.eye(dim, dtype=np.cfloat), 2, axis=1)

    elif target_name == "shift_half":
        _logger.info(
            "%s a unit matrix with diagonal cyclically "
            "shifted dim/2 positions to the right (half of matrix size)",
            msg,
        )
        target = np.roll(np.eye(dim, dtype=np.cfloat), dim // 2, axis=1)

    elif target_name == "random_perm":
        _logger.info("%s a randomly permuted identity matrix", msg)
        target = np.take(np.eye(dim, dtype=np.cfloat), np.random.permutation(dim), axis=1)

    else:
        raise ValueError(
            f"target type is not in the set of supported ones: "
            f"{available_target_matrix_types()}, got {target_name}",
        )

    # Check the matrix is really a unitary one.
    if num_qubits <= 8:
        tol = float(np.sqrt(np.finfo(np.float64).eps))
        if not np.allclose(np.vdot(target, target), dim, atol=tol, rtol=tol):
            raise ValueError("target matrix seems not a unitary one")

    toc = perf_counter()
    _logger.info("Target matrix has been prepared in %0.2f secs", toc - tic)
    return target


def make_su_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Creates SU matrix from the input unitary one via multiplication
    by a complex number.

    Args:
        mat: input unitary matrix.

    Returns:
        a new SU matrix, generated from the input one, or the same input matrix,
        if it is already in SU class.
    """
    assert chk.complex_2d(mat)
    tol = float(np.sqrt(np.finfo(float).eps))
    dim = mat.shape[0]
    det = np.linalg.det(mat)
    if not np.isclose(det, 1.0, atol=tol, rtol=tol):
        mat = mat / np.power(det, (1.0 / dim))  # a new matrix
        _logger.info("the target U matrix has been converted into SU one")
    return mat
