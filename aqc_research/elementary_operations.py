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
Elementary functions required by AQC parametric circuit.
"""

import cmath
import numpy as np
import aqc_research.checking as chk

# Real and complex types we want to use with PyTorch: either float32 or float64.
# _glo_th_real_t = th.float64
# _glo_th_complex_t = th.complex64 if _glo_th_real_t == th.float32 else th.complex128

# Identity gate acting on a single qubit:
_glo_eye2x2 = np.array([[1, 0], [0, 1]], dtype=np.cfloat)

# Projectors |0><0| and |1><1| respectively:
_glo_np00 = np.array([[1, 0], [0, 0]], dtype=np.cfloat)
_glo_np11 = np.array([[0, 0], [0, 1]], dtype=np.cfloat)
# _glo_th00 = th.tensor([[1, 0], [0, 0]], dtype=_glo_th_complex_t)
# _glo_th11 = th.tensor([[0, 0], [0, 1]], dtype=_glo_th_complex_t)

###############################################################################
# Numpy operations.
###############################################################################


def np_block_matrix(
    n: int, c: int, t: int, c_mat: np.ndarray, t_mat: np.ndarray, gate_mat: np.ndarray
) -> np.ndarray:
    """
    Creates unitary matrix of a unit block placed at ``c`` and ``t``.
                            _______
        control -----o-----| c_mat |-----
                     |      -------
                     |      _______
        target  ----|G|----| t_mat |-----
                            -------
    Args:
        n: number of qubits.
        c: index of control qubit.
        t: index of target qubit.
        c_mat: 1-qubit matrix applied at control qubit.
        t_mat: 1-qubit matrix applied at target qubit.
        gate_mat: 1-qubit controlled gate (G) of entangler.

    Returns:
        n-qubits unitary of a unit block placed at ``c`` and ``t``.
    """
    c00 = c_mat @ _glo_np00
    c11 = c_mat @ _glo_np11
    t_g = t_mat @ gate_mat

    if c < t:
        eye1 = np.eye(2**c, dtype=np.cfloat)
        eye2 = np.eye(2 ** (t - c - 1), dtype=np.cfloat)
        eye3 = np.eye(2 ** (n - 1 - t), dtype=np.cfloat)

        mat1 = np.kron(np.kron(np.kron(np.kron(eye1, c00), eye2), t_mat), eye3)
        mat2 = np.kron(np.kron(np.kron(np.kron(eye1, c11), eye2), t_g), eye3)
    else:
        eye1 = np.eye(2**t, dtype=np.cfloat)
        eye2 = np.eye(2 ** (c - t - 1), dtype=np.cfloat)
        eye3 = np.eye(2 ** (n - 1 - c), dtype=np.cfloat)

        mat1 = np.kron(np.kron(np.kron(np.kron(eye1, t_mat), eye2), c00), eye3)
        mat2 = np.kron(np.kron(np.kron(np.kron(eye1, t_g), eye2), c11), eye3)

    mat1 += mat2
    return mat1


def np_cx_matrix(n: int, c: int, t: int) -> np.ndarray:
    """
    Creates unitary matrix of a CX gate placed at ``c`` and ``t``.
    This function is useful for generating states using CNOT-only circuit.

        control -----o-----
                     |
                     |
        target  ----|X|----

    Args:
        n: number of qubits.
        c: index of control qubit.
        t: index of target qubit.

    Returns:
        n-qubits unitary of a unit block placed at ``c`` and ``t``.
    """
    iden1q = _glo_eye2x2  # identity matrix acting on one qubit

    if c < t:
        eye1 = np.eye(2**c, dtype=np.cfloat)
        eye2 = np.eye(2 ** (t - c - 1), dtype=np.cfloat)
        eye3 = np.eye(2 ** (n - 1 - t), dtype=np.cfloat)

        mat1 = np.kron(np.kron(np.kron(np.kron(eye1, _glo_np00), eye2), iden1q), eye3)
        mat2 = np.kron(np.kron(np.kron(np.kron(eye1, _glo_np11), eye2), np_x()), eye3)
    else:
        eye1 = np.eye(2**t, dtype=np.cfloat)
        eye2 = np.eye(2 ** (c - t - 1), dtype=np.cfloat)
        eye3 = np.eye(2 ** (n - 1 - c), dtype=np.cfloat)

        mat1 = np.kron(np.kron(np.kron(np.kron(eye1, iden1q), eye2), _glo_np00), eye3)
        mat2 = np.kron(np.kron(np.kron(np.kron(eye1, np_x()), eye2), _glo_np11), eye3)

    mat1 += mat2
    return mat1


def np_rx(phi: float) -> np.ndarray:
    """
    Computes an RX rotation by the angle of ``phi``.

    Args:
        phi: rotation angle.

    Returns:
        an RX rotation matrix.
    """
    out = np.zeros((2, 2), dtype=np.cfloat)
    a = 0.5 * phi
    cs, sn = cmath.cos(a), -1j * cmath.sin(a)
    out[0, 0] = cs
    out[0, 1] = sn
    out[1, 0] = sn
    out[1, 1] = cs
    return out


def make_rx(phi: float, out: np.ndarray) -> np.ndarray:
    """
    Makes a 2x2 matrix that corresponds to X-rotation gate.
    This implementation outputs result into external matrix, and it is faster
    than initialization from a list.

    Args:
        phi: rotation angle.
        out: placeholder for the result (2x2, complex-valued matrix).

    Returns:
        rotation gate, same object as referenced by "out".
    """
    chk.is_float(phi)
    chk.complex_2d(out, out.shape == (2, 2))

    a = 0.5 * phi
    cs, sn = cmath.cos(a), -1j * cmath.sin(a)
    out[0, 0] = cs
    out[0, 1] = sn
    out[1, 0] = sn
    out[1, 1] = cs
    return out


def np_ry(phi: float) -> np.ndarray:
    """
    Computes an RY rotation by the angle of ``phi``.

    Args:
        phi: rotation angle.

    Returns:
        an RY rotation matrix.
    """
    out = np.zeros((2, 2), dtype=np.cfloat)
    a = 0.5 * phi
    cs, sn = cmath.cos(a), cmath.sin(a)
    out[0, 0] = cs
    out[0, 1] = -sn
    out[1, 0] = sn
    out[1, 1] = cs
    return out


def make_ry(phi: float, out: np.ndarray) -> np.ndarray:
    """
    Makes a 2x2 matrix that corresponds to Y-rotation gate.
    This implementation outputs result into external matrix, and it is faster
    than initialization from a list.

    Args:
        phi: rotation angle.
        out: placeholder for the result (2x2, complex-valued matrix).

    Returns:
        rotation gate, same object as referenced by "out".
    """
    chk.is_float(phi)
    chk.complex_2d(out, out.shape == (2, 2))

    a = 0.5 * phi
    cs, sn = cmath.cos(a), cmath.sin(a)
    out[0, 0] = cs
    out[0, 1] = -sn
    out[1, 0] = sn
    out[1, 1] = cs
    return out


def np_rz(phi: float) -> np.ndarray:
    """
    Computes an RZ rotation by the angle of ``phi``.

    Args:
        phi: rotation angle.

    Returns:
        an RZ rotation matrix.
    """
    out = np.zeros((2, 2), dtype=np.cfloat)
    exp = cmath.exp(0.5j * phi)
    out[0, 0] = 1.0 / exp
    out[1, 1] = exp
    return out


def make_rz(phi: float, out: np.ndarray) -> np.ndarray:
    """
    Makes a 2x2 matrix that corresponds to Z-rotation gate.
    This implementation outputs result into external matrix, and it is faster
    than initialization from a list.

    Args:
        phi: rotation angle.
        out: placeholder for the result (2x2, complex-valued matrix).

    Returns:
        rotation gate, same object as referenced by "out".
    """
    chk.is_float(phi)
    chk.complex_2d(out, out.shape == (2, 2))

    exp = cmath.exp(0.5j * phi)
    out[0, 0] = 1.0 / exp
    out[0, 1] = 0
    out[1, 0] = 0
    out[1, 1] = exp
    return out


def np_phase(phi: float) -> np.ndarray:
    """
    Computes a phase gate with parameter ``phi``.

    Args:
        phi: phase parameter.

    Returns:
        phase gate matrix.
    """
    out = np.eye(2, dtype=np.cfloat)
    out[1, 1] = cmath.exp(1j * phi)
    return out


def np_x() -> np.ndarray:
    """
    Computes an X gate.

    Returns:
        phase gate matrix.
    """
    out = np.zeros((2, 2), dtype=np.cfloat)
    out[0, 1] = 1
    out[1, 0] = 1
    return out


def np_z() -> np.ndarray:
    """
    Computes an Z gate.

    Returns:
        phase gate matrix.
    """
    out = np.eye(2, dtype=np.cfloat)
    out[1, 1] = -1
    return out
