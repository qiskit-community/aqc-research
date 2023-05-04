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
Utility functions for debugging and testing.
"""

from typing import Tuple
import numpy as np
from scipy.stats import unitary_group
import aqc_research.checking as chk


def relative_diff(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """
    Computes relative residual either between two matrices in Frobenius norm
    or between two vectors in L2 norm. The second argument is considered as
    the desired (aka ground truth) target (its norm divides the residual).
    """
    tiny = np.finfo(np.float64).eps ** 2
    assert isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)
    assert arr1.shape == arr2.shape

    if arr1.ndim == 2:
        norm = float(np.linalg.norm(arr2, "fro"))
        return float(np.linalg.norm(arr1 - arr2, "fro")) / max(norm, tiny)

    if arr1.ndim == 1:
        norm = float(np.linalg.norm(arr2))
        return float(np.linalg.norm(arr1 - arr2)) / max(norm, tiny)

    if arr1.ndim == 0:  # scalars
        return float((np.abs(arr1 - arr2)) / max(np.abs(arr1), np.abs(arr2), tiny))

    raise ValueError("expects matrices, vectors or scalars as an input")


def hs_product(arr1: np.ndarray, arr2: np.ndarray) -> np.cfloat:
    """
    Computes Hilbert-Schmidt product between two vectors or matrices.
    In case of matrix inputs, we normalize the product such that its absolute
    value is less than or equal to unit one for unitary matrices.
    """
    assert isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray)
    assert arr1.shape == arr2.shape
    assert arr1.ndim == 1 or (arr1.ndim == 2 and arr1.shape[0] == arr1.shape[1])
    scale = float(arr1.shape[0]) if arr1.ndim == 2 else 1.0
    return np.cfloat(np.vdot(arr1, arr2).item() / scale)


class RelativeDiff:
    """Accumulator of the maximum relative difference between two arrays."""

    def __init__(self):
        self._max_diff = 0.0

    def update(self, arr1: np.ndarray, arr2: np.ndarray):
        """Updates the maximum relative difference."""
        self._max_diff = max(self._max_diff, relative_diff(np.asarray(arr1), np.asarray(arr2)))

    def max_diff(self) -> str:
        """Returns a string with accumulated maximum relative difference."""
        return f"Max. relative difference: {self._max_diff}"


def eye_int(num_qubits: int) -> np.ndarray:
    """
    Creates an identity matrix with integer entries.

    Args:
        num_qubits: number of qubits.

    Returns:
        unit matrix of size ``2^n`` with integer entries.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    return np.eye(2**num_qubits, dtype=np.int64)


def kron3(a_mat: np.ndarray, b_mat: np.ndarray, c_mat: np.ndarray) -> np.ndarray:
    """
    Computes Kronecker product of 3 matrices.
    """
    return np.kron(a_mat, np.kron(b_mat, c_mat))


def kron5(
    a_mat: np.ndarray,
    b_mat: np.ndarray,
    c_mat: np.ndarray,
    d_mat: np.ndarray,
    e_mat: np.ndarray,
) -> np.ndarray:
    """
    Computes Kronecker product of 5 matrices.
    """
    return np.kron(np.kron(np.kron(np.kron(a_mat, b_mat), c_mat), d_mat), e_mat)


def rand_circuit(num_qubits: int, depth: int) -> np.ndarray:
    """
    Generates a random circuit of unit blocks for debugging and testing.
    """
    blocks = np.tile(np.arange(num_qubits).reshape(num_qubits, 1), depth)
    for i in range(depth):
        np.random.shuffle(blocks[:, i])
    return blocks[0:2, :].copy()


def rand_vec(dim: int, unit: bool = False) -> np.ndarray:
    """
    Returns random vector of complex values.

    Args:
        dim: vector size.
        unit: flag enables normalization of generated vector.
    Returns:
        a random vector; normalized if ``unit`` is True.
    """
    assert chk.is_int(dim, dim > 0) and isinstance(unit, bool)
    vec = np.random.rand(dim).astype(np.cfloat) + np.random.rand(dim).astype(np.cfloat) * 1j
    if unit:
        vec /= max(np.linalg.norm(vec), np.finfo(np.float64).eps ** 2)
    return vec


def rand_mat(dim: int, kind: str = "complex") -> np.ndarray:
    """
    Generates a random complex or integer value matrix.

    Args:
        dim: matrix size dim-x-dim.
        kind: one of {"complex", "randint"}.
    Returns:
        a random matrix.
    """
    assert chk.is_int(dim, dim > 0)
    assert chk.is_str(kind, kind in ["complex", "randint"])
    if kind == "complex":
        return (
            np.random.rand(dim, dim).astype(np.cfloat)
            + np.random.rand(dim, dim).astype(np.cfloat) * 1j
        )

    return np.random.randint(low=1, high=100, size=(dim, dim), dtype=np.int64)


def rand_rect_mat(nrows: int, ncols: int, kind: str = "complex") -> np.ndarray:
    """
    Generates a random rectangular complex or integer value matrix.

    Args:
        nrows: number of rows.
        ncols: number of columns.
        kind: one of {"complex", "randint"}.
    Returns:
        a random matrix.
    """
    assert chk.is_int(nrows, nrows > 0)
    assert chk.is_int(ncols, ncols > 0)
    assert chk.is_str(kind, kind in ["complex", "randint"])
    if kind == "complex":
        mat = np.random.rand(nrows, ncols) + 1j * np.random.rand(nrows, ncols)
        assert mat.dtype == np.complex128
        return mat

    return np.random.randint(low=1, high=100, size=(nrows, ncols), dtype=np.int64)


def rand_su_mat(dim: int) -> np.ndarray:
    """
    Generates a random SU matrix.

    Args:
        dim: matrix size dim-x-dim.

    Returns
        random SU matrix.
    """
    tol = float(np.sqrt(np.finfo(np.float64).eps))
    assert chk.is_int(dim, dim >= 2)
    u_mat = unitary_group.rvs(dim)
    u_mat /= np.linalg.det(u_mat) ** (1.0 / float(dim))
    assert chk.complex_2d_square(u_mat)
    assert abs(np.linalg.det(u_mat) - 1.0) < tol
    assert np.allclose(np.vdot(u_mat, u_mat), 1.0, atol=tol, rtol=tol)
    return u_mat


def test_mat2x2(
    num_qubits: int, position: int, kind: str = "complex"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a ``2^n x 2^n`` (random) matrix made as a Kronecker product of
    identity ones and a single 1-qubit gate. This models a layer in quantum
    circuit with an arbitrary 1-qubit gate somewhere in the middle.
    **Important**:
    Here we adopt indexing usual in quantum computing: the most significant
    bit has index 0, the least significant one has index num_qubits-1.

    ``I_0 kron I_1 kron ... kron G_{position} kron ... kron I_{n-1}``,
    where the subscript enumerates bits, and the bit 0 is located at the top-left
    corner of a matrix (or vector).

    Args:
        num_qubits: number of qubits.
        position: index of qubit a 1-qubit gate is acting on.
        kind: one of {"complex", "primes", "randint"}.
    Returns:
        (1) 2^n x 2^n random matrix;
        (2) 2 x 2 matrix of 1-qubit gate used for matrix construction.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2) and chk.is_int(position)
    assert 0 <= position < num_qubits
    assert chk.is_str(kind, kind in ["complex", "primes", "randint"])

    if kind == "primes":
        g_mat = np.array([[2, 3], [5, 7]], dtype=np.int64)
    else:
        g_mat = rand_mat(dim=2, kind=kind)
    m_mat = kron3(eye_int(position), g_mat, eye_int(num_qubits - position - 1))
    return m_mat, g_mat


def test_mat4x4(
    num_qubits: int, j_position: int, k_position: int, kind: str = "complex"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a ``2^n x 2^n`` (random) matrix made as a Kronecker product of
    identity ones and a single 2-qubit gate. This models a layer in quantum
    circuit with an arbitrary 2-qubit gate somewhere in the middle.
    **Important**:
    Here we adopt indexing usual in quantum computing: the most significant
    bit has index 0, the least significant one has index num_qubits-1.

    Args:
        num_qubits: number of qubits.
        j_position: index of the 1st (control) qubit the 2-qubit gate acting on.
        k_position: index of the 2nd (target) qubit the 2-qubit gate acting on.
        kind: one of {"complex", "primes", "randint"}.
    Returns:
        (1) 2^n x 2^n random matrix;
        (2) 4 x 4 matrix of 2-qubit gate used for matrix construction.
    """
    n_q, ctrl, targ = num_qubits, j_position, k_position  # shorthand aliases
    assert chk.is_int(n_q) and chk.is_int(ctrl) and chk.is_int(targ)
    assert ctrl != targ and 0 <= ctrl < n_q and 0 <= targ < n_q and 2 <= n_q
    assert isinstance(kind, str) and (kind in {"complex", "primes", "randint"})

    if kind == "primes":
        a_mat = np.array([[2, 3], [5, 7]], dtype=np.int64)
        b_mat = np.array([[11, 13], [17, 19]], dtype=np.int64)
        c_mat = np.array([[47, 53], [41, 43]], dtype=np.int64)
        d_mat = np.array([[31, 37], [23, 29]], dtype=np.int64)
    else:
        a_mat = rand_mat(dim=2, kind=kind)
        b_mat = rand_mat(dim=2, kind=kind)
        c_mat = rand_mat(dim=2, kind=kind)
        d_mat = rand_mat(dim=2, kind=kind)

    if ctrl < targ:
        m_mat = kron5(
            eye_int(ctrl),
            a_mat,
            eye_int(targ - ctrl - 1),
            b_mat,
            eye_int(n_q - targ - 1),
        ) + kron5(
            eye_int(ctrl),
            c_mat,
            eye_int(targ - ctrl - 1),
            d_mat,
            eye_int(n_q - targ - 1),
        )
    else:  # targ < ctrl
        m_mat = kron5(
            eye_int(targ),
            b_mat,
            eye_int(ctrl - targ - 1),
            a_mat,
            eye_int(n_q - ctrl - 1),
        ) + kron5(
            eye_int(targ),
            d_mat,
            eye_int(ctrl - targ - 1),
            c_mat,
            eye_int(n_q - ctrl - 1),
        )

    g_mat = np.kron(a_mat, b_mat) + np.kron(c_mat, d_mat)
    return m_mat, g_mat
