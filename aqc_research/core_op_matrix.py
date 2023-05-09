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
Core operations for approximating quantum compiling, where gate and circuit
matrices are multiplied by the arbitrary, right-hand side, rectangular matrices.

We exploit sparsity patterns of the gate matrices avoiding their explicit
construction as full size ``2^n x 2^n`` ones. This saves memory and improves
performance.

**Note**, all routines in this module adopt Qiskit bit ordering.
"""


from typing import Optional
import numpy as np
import aqc_research.checking as chk
from aqc_research.parametric_circuit import ParametricCircuit
from aqc_research.elementary_operations import make_rx, make_ry, make_rz


def rx_mul_mat(
    angle: float,
    qubit_no: int,
    mat: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Computes the product of Rx-gate matrix times right-hand side matrix.
    **Note**, the matrix ``mat`` will be modified in-place.

    Args:
        angle: angular parameter of the gate.
        qubit_no: position of a qubit where 2x2 gate is applied.
        mat: right-hand side matrix to be multiplied in-place.
        workspace: workspace of size greater or equal to the size of ``mat``.

    Returns:
        input matrix modified in-place.
    """
    dim = mat.shape[0]
    assert chk.is_float(angle)
    assert chk.is_int(qubit_no, 0 <= qubit_no < int(round(np.log2(dim))))
    assert chk.complex_2d(mat, mat.shape[1] <= dim)
    assert chk.complex_array(workspace, workspace.size >= mat.size)
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**qubit_no)))
    w = workspace.ravel()[0 : mat.size].reshape(m.shape)
    h = m.shape[1] // 2  # half number of columns of reshaped matrix
    np.multiply(m, -1j * np.sin(0.5 * angle), out=w)  # w = m * (-i) * sin(a/2)
    m *= np.cos(0.5 * angle)
    m[:, :h] += w[:, h:]
    m[:, h:] += w[:, :h]
    return mat


def ry_mul_mat(
    angle: float,
    qubit_no: int,
    mat: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Computes the product of Ry-gate matrix times right-hand side matrix.
    **Note**, the matrix ``mat`` will be modified in-place.

    Args:
        angle: angular parameter of the gate.
        qubit_no: position of a qubit where 2x2 gate is applied.
        mat: right-hand side matrix to be multiplied in-place.
        workspace: workspace of size greater or equal to the size of ``mat``.

    Returns:
        input matrix modified in-place.
    """
    dim = mat.shape[0]
    assert chk.is_float(angle)
    assert chk.is_int(qubit_no, 0 <= qubit_no < int(round(np.log2(dim))))
    assert chk.complex_2d(mat, mat.shape[1] <= dim)
    assert chk.complex_array(workspace, workspace.size >= mat.size)
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**qubit_no)))
    w = workspace.ravel()[0 : mat.size].reshape(m.shape)
    h = m.shape[1] // 2  # half number of columns of reshaped matrix
    np.multiply(m, np.sin(0.5 * angle), out=w)  # w = m * sin(a/2)
    m *= np.cos(0.5 * angle)
    m[:, :h] -= w[:, h:]
    m[:, h:] += w[:, :h]
    return mat


def rz_mul_mat(
    angle: float,
    qubit_no: int,
    mat: np.ndarray,
    ___: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes the product of Rz-gate matrix times right-hand side matrix.
    **Note**, the matrix ``mat`` will be modified in-place.

    Args:
        angle: angular parameter of the gate.
        qubit_no: position of a qubit where 2x2 gate is applied.
        mat: right-hand side matrix to be multiplied in-place.
        ___: argument retained for interface compatibility.

    Returns:
        input matrix modified in-place.
    """
    dim = mat.shape[0]
    assert chk.is_float(angle)
    assert chk.is_int(qubit_no, 0 <= qubit_no < int(round(np.log2(dim))))
    assert chk.complex_2d(mat, mat.shape[1] <= dim)
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**qubit_no)))
    h = m.shape[1] // 2  # half number of columns of reshaped matrix
    m[:, :h] *= np.exp(-0.5j * angle)
    m[:, h:] *= np.exp(+0.5j * angle)
    return mat


def cx_mul_mat(
    ctrl: int,
    targ: int,
    ___: float,
    mat: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Multiplies specified matrix on the left by (expanded) CX gate matrix.
    **Note**, the matrix will be modified in-place.

    Args:
        ctrl: index of control qubit.
        targ: index of target qubit.
        ___: argument retained for interface compatibility.
        mat: right-hand side matrix to be multiplied by the gate one.
        workspace: workspace of size greater or equal to the size of ``mat``.

    Returns:
        input mat modified in-place.
    """
    dim = mat.shape[0]
    n = int(round(np.log2(dim)))  # number of qubits
    assert 0 <= ctrl < n and 0 <= targ < n and ctrl != targ
    assert chk.complex_2d(mat, mat.shape[1] <= dim)
    assert chk.complex_array(workspace, workspace.size >= mat.size)
    assert not np.may_share_memory(mat, workspace)

    # Reshape at control qubit.
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**ctrl)))
    w = workspace.ravel()[0 : mat.size].reshape(m.shape)
    h = m.shape[1] // 2  # half number of columns of reshaped matrix

    # w = full_matrix(|1><1|_c) @ mat.
    w[:, :h] = 0
    w[:, h:] = m[:, h:]

    # m = full_matrix(|0><0|_c kron I_t) @ mat.
    m[:, h:] = 0

    # Reshape at target qubit.
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**targ)))
    w = workspace.ravel()[0 : mat.size].reshape(m.shape)
    h = m.shape[1] // 2  # half number of columns of reshaped matrix

    # mat = mat + full_matrix(|1><1|_c @ X_t) @ mat.
    m[:, :h] += w[:, h:]
    m[:, h:] += w[:, :h]
    return mat


def cz_mul_mat(
    ctrl: int,
    targ: int,
    ___: float,
    mat: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Multiplies specified matrix on the left by (expanded) CZ gate matrix.
    **Note**, the matrix will be modified in-place.

    Args:
        ctrl: index of control qubit.
        targ: index of target qubit.
        ___: argument retained for interface compatibility.
        mat: right-hand side matrix to be multiplied by the gate one.
        workspace: workspace of size greater or equal to the size of ``mat``.

    Returns:
        input mat modified in-place.
    """
    dim = mat.shape[0]
    n = int(round(np.log2(dim)))  # number of qubits
    assert 0 <= ctrl < n and 0 <= targ < n and ctrl != targ
    assert chk.complex_2d(mat, mat.shape[1] <= dim)
    assert chk.complex_array(workspace, workspace.size >= mat.size)
    assert not np.may_share_memory(mat, workspace)

    # Reshape at control qubit.
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**ctrl)))
    w = workspace.ravel()[0 : mat.size].reshape(m.shape)
    h = m.shape[1] // 2  # half number of columns of reshaped matrix

    # w = full_matrix(|1><1|_c) @ mat.
    w[:, :h] = 0
    w[:, h:] = m[:, h:]

    # m = full_matrix(|0><0|_c kron I_t) @ mat.
    m[:, h:] = 0

    # Reshape at target qubit.
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**targ)))
    w = workspace.ravel()[0 : mat.size].reshape(m.shape)
    h = m.shape[1] // 2  # half number of columns of reshaped matrix

    # mat = mat + full_matrix(|1><1|_c @ Z_t) @ mat.
    m[:, :h] += w[:, :h]
    m[:, h:] -= w[:, h:]
    return mat


def cp_mul_mat(
    ctrl: int,
    targ: int,
    angle: float,
    mat: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Multiplies specified matrix on the left by (expanded) CPhase gate matrix.
    **Note**, the matrix will be modified in-place.

    Args:
        ctrl: index of control qubit.
        targ: index of target qubit.
        angle: parameter of the controlled-phase gate.
        mat: right-hand side matrix to be multiplied by the gate one.
        workspace: workspace of size greater or equal to the size of ``mat``.

    Returns:
        input mat modified in-place.
    """
    dim = mat.shape[0]
    n = int(round(np.log2(dim)))  # number of qubits
    assert 0 <= ctrl < n and 0 <= targ < n and ctrl != targ
    assert chk.is_float(angle)
    assert chk.complex_2d(mat, mat.shape[1] <= dim)
    assert chk.complex_array(workspace, workspace.size >= mat.size)
    assert not np.may_share_memory(mat, workspace)

    # Reshape at control qubit.
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**ctrl)))
    w = workspace.ravel()[0 : mat.size].reshape(m.shape)
    h = m.shape[1] // 2  # half number of columns of reshaped matrix

    # w = full_matrix(|1><1|_c) @ mat.
    w[:, :h] = 0
    w[:, h:] = m[:, h:]

    # m = full_matrix(|0><0|_c kron I_t) @ mat.
    m[:, h:] = 0

    # Reshape at target qubit.
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**targ)))
    w = workspace.ravel()[0 : mat.size].reshape(m.shape)
    h = m.shape[1] // 2  # half number of columns of reshaped matrix

    # mat = mat + full_matrix(|1><1|_c @ P_t) @ mat.
    w[:, h:] *= np.exp(1j * angle)
    m += w
    return mat


def x_dot_mat(
    qubit_no: int, w_mat: np.ndarray, z_mat: np.ndarray, workspace: np.ndarray
) -> np.complex128:
    """
    Computes 0.5j * <X@w|z>.
    Important: vdot() is the most effective on contiguous arrays. This is why
    we first form a vector |gate @ w> and then apply vdot(). Although,
    a temporary storage can be avoided in principle, this approach is faster.

    Args:
        qubit_no: position of a qubit where 2x2 gate is applied.
        w_mat: w matrix.
        z_mat: z matrix.
        workspace: workspace of size greater or equal to the size of matrices.

    Returns:
        0.5j * <X@w|z>.
    """
    dim = w_mat.shape[0]
    assert chk.is_int(qubit_no, 0 <= qubit_no < int(round(np.log2(dim))))
    assert chk.complex_2d(w_mat, w_mat.shape[1] <= dim)
    assert chk.complex_2d(z_mat, z_mat.shape[1] <= dim)
    assert w_mat.shape == z_mat.shape
    assert chk.complex_array(workspace, workspace.size >= w_mat.size)
    assert not np.may_share_memory(w_mat, workspace)

    # Reshape at qubit position.
    s_mat = workspace.ravel()[0 : w_mat.size].reshape(w_mat.shape)
    w = w_mat.reshape((-1, 2 * w_mat.shape[1] * (2**qubit_no)))
    s = s_mat.reshape((-1, 2 * w_mat.shape[1] * (2**qubit_no)))
    h = w.shape[1] // 2  # half number of columns of reshaped matrix
    s[:, :h] = w[:, h:]
    s[:, h:] = w[:, :h]
    return 0.5j * np.vdot(s_mat, z_mat)


def y_dot_mat(
    qubit_no: int, w_mat: np.ndarray, z_mat: np.ndarray, workspace: np.ndarray
) -> np.complex128:
    """
    Computes 0.5j * <Y@w|z>.
    Important: vdot() is the most effective on contiguous arrays. This is why
    we first form a vector |gate @ w> and then apply vdot(). Although,
    a temporary storage can be avoided in principle, this approach is faster.

    Args:
        qubit_no: position of a qubit where 2x2 gate is applied.
        w_mat: w matrix.
        z_mat: z matrix.
        workspace: workspace of size greater or equal to the size of matrices.

    Returns:
        0.5j * <X@w|z>.
    """
    dim = w_mat.shape[0]
    assert chk.is_int(qubit_no, 0 <= qubit_no < int(round(np.log2(dim))))
    assert chk.complex_2d(w_mat, w_mat.shape[1] <= dim)
    assert chk.complex_2d(z_mat, z_mat.shape[1] <= dim)
    assert w_mat.shape == z_mat.shape
    assert chk.complex_array(workspace, workspace.size >= w_mat.size)
    assert not np.may_share_memory(w_mat, workspace)

    # Reshape at qubit position.
    s_mat = workspace.ravel()[0 : w_mat.size].reshape(w_mat.shape)
    w = w_mat.reshape((-1, 2 * w_mat.shape[1] * (2**qubit_no)))
    s = s_mat.reshape((-1, 2 * w_mat.shape[1] * (2**qubit_no)))
    h = w.shape[1] // 2  # half number of columns of reshaped matrix
    np.negative(w[:, h:], out=s[:, :h])
    s[:, h:] = w[:, :h]
    return 0.5 * np.vdot(s_mat, z_mat)  # 0.5 = 0.5j * conj(j)


def z_dot_mat(
    qubit_no: int, w_mat: np.ndarray, z_mat: np.ndarray, workspace: np.ndarray
) -> np.complex128:
    """
    Computes 0.5j * <X@w|z>.
    Important: vdot() is the most effective on contiguous arrays. This is why
    we first form a vector |gate @ w> and then apply vdot(). Although,
    a temporary storage can be avoided in principle, this approach is faster.

    Args:
        qubit_no: position of a qubit where 2x2 gate is applied.
        w_mat: w matrix.
        z_mat: y matrix.
        workspace: workspace of size greater or equal to the size of matrices.

    Returns:
        0.5j * <X@w|z>.
    """
    dim = w_mat.shape[0]
    assert chk.is_int(qubit_no, 0 <= qubit_no < int(round(np.log2(dim))))
    assert chk.complex_2d(w_mat, w_mat.shape[1] <= dim)
    assert chk.complex_2d(z_mat, z_mat.shape[1] <= dim)
    assert w_mat.shape == z_mat.shape
    assert chk.complex_array(workspace, workspace.size >= w_mat.size)
    assert not np.may_share_memory(w_mat, workspace)

    # Reshape at qubit position.
    s_mat = workspace.ravel()[0 : w_mat.size].reshape(w_mat.shape)
    w = w_mat.reshape((-1, 2 * w_mat.shape[1] * (2**qubit_no)))
    s = s_mat.reshape((-1, 2 * w_mat.shape[1] * (2**qubit_no)))
    h = w.shape[1] // 2  # half number of columns of reshaped matrix
    s[:] = w[:]  # faster than just:  s[:, :h] = w[:, :h]
    np.negative(w[:, h:], out=s[:, h:])
    return 0.5j * np.vdot(s_mat, z_mat)


def gate2x2_mul_mat(
    qubit_no: int,
    gate2x2: np.ndarray,
    mat: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Computes the product of 2x2 gate matrix (expanded to the full size) and
    the right-hand side, arbitrary matrix: ``mat <- gate @ mat``.

    Args:
        qubit_no: position of a qubit where 2x2 gate is applied.
        gate2x2: ``2x2`` gate matrix.
        mat: right-hand side matrix to be multiplied in-place.
        workspace: temporary array of shape (2, vec.size).

    Returns:
        input matrix ``mat`` multiplied by block matrix in-place.
    """
    dim = mat.shape[0]
    assert 0 <= qubit_no < int(round(np.log2(dim)))
    assert chk.complex_or_float_2d(gate2x2, gate2x2.shape == (2, 2))
    assert chk.complex_2d(mat, mat.shape[1] <= dim)
    assert chk.complex_array(workspace, workspace.size >= mat.size)
    assert not np.may_share_memory(mat, workspace)

    # Reshape at qubit position.
    m = mat.reshape((-1, 2 * mat.shape[1] * (2**qubit_no)))
    w = workspace.ravel()[0 : mat.size].reshape(m.shape)
    h = m.shape[1] // 2  # half number of columns of reshaped matrix
    np.multiply(m[:, h:], gate2x2[0, 1], w[:, :h])
    np.multiply(m[:, :h], gate2x2[1, 0], w[:, h:])
    m[:, :h] *= gate2x2[0, 0]
    m[:, h:] *= gate2x2[1, 1]
    m += w
    return mat


def derv_cphase(
    ctrl: int,
    targ: int,
    w_mat: np.ndarray,
    z_mat: np.ndarray,
    workspace: np.ndarray,
) -> np.complex128:
    """
    Computes derivative of <w|z> by CPhase gate parameter.

    Args:
        ctrl: index of control qubit.
        targ: index of target qubit.
        w_mat: w matrix.
        z_mat: z matrix.
        workspace: workspace of size greater or equal to the size of matrices.

    Returns:
        derivative of <w|z> by CPhase gate parameter.
    """
    dim = w_mat.shape[0]
    n = int(round(np.log2(dim)))  # number of qubits
    assert 0 <= ctrl < n and 0 <= targ < n and ctrl != targ
    assert chk.complex_2d(w_mat, w_mat.shape[1] <= dim)
    assert chk.complex_2d(z_mat, z_mat.shape[1] <= dim)
    assert w_mat.shape == z_mat.shape
    assert chk.complex_array(workspace, workspace.size >= w_mat.size)
    assert not np.may_share_memory(w_mat, workspace)

    # Reshape at control qubit.
    s_mat = workspace.ravel()[0 : w_mat.size].reshape(w_mat.shape)
    w = w_mat.reshape((-1, 2 * w_mat.shape[1] * (2**ctrl)))
    s = s_mat.reshape((-1, 2 * w_mat.shape[1] * (2**ctrl)))
    h = w.shape[1] // 2  # half number of columns of reshaped matrix

    # s = full_matrix(|1><1|_c) @ w_mat.
    s[:, :h] = 0
    s[:, h:] = w[:, h:]

    # Reshape at target qubit.
    s_mat = workspace.ravel()[0 : w_mat.size].reshape(w_mat.shape)
    w = w_mat.reshape((-1, 2 * w_mat.shape[1] * (2**targ)))
    s = s_mat.reshape((-1, 2 * w_mat.shape[1] * (2**targ)))
    h = w.shape[1] // 2  # half number of columns of reshaped matrix

    # s = full_matrix(|1><1|_t) @ s.
    s[:, :h] = 0
    return -1j * np.vdot(s_mat, z_mat)


def v_mul_mat(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    mat: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Multiplies circuit matrix by the right-hand side matrix: ``mat <- V @ mat``.
    **Note**, the input matrix will be modified in-place.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of ansatz parametric circuit.
        mat: right-hand side matrix.
        workspace: workspace of size greater or equal to the size of ``mat``.

    Returns:
        modified input matrix ``mat``.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
    assert chk.complex_2d(mat, mat.shape[0] >= mat.shape[1])
    assert chk.complex_array(workspace, workspace.size >= mat.size)
    assert not np.may_share_memory(mat, workspace)
    assert mat.data.contiguous  # !!!

    blocks = circ.blocks
    mat2 = np.zeros((2, 2, 2), dtype=np.cfloat)  # temporary 2x2 matrices

    # Convenient split into parameter sub-sets for 1- and 2-qubit gates.
    thetas1q = circ.subset1q(thetas)
    thetas2q = circ.subset2q(thetas)

    # Entangling gates are: CX, CZ or CP.
    if circ.entangler == "cp":
        cg_mul_mat = cp_mul_mat  # function: controlled-gate times matrix
        make_rs = make_rz  # swappable with entangling gate

        def _get_angle(_th: np.ndarray) -> float:
            return float(_th[4])

    elif circ.entangler == "cz":
        cg_mul_mat = cz_mul_mat  # function: controlled-gate times matrix
        make_rs = make_rz  # swappable with entangling gate

        def _get_angle(__: np.ndarray) -> float:
            return float(0)

    else:  # cx
        cg_mul_mat = cx_mul_mat  # function: controlled-gate times matrix
        make_rs = make_rx  # swappable with entangling gate

        def _get_angle(__: np.ndarray) -> float:
            return float(0)

    # Multiply vector by the initial layer of rotation gates Rz @ Ry @ Rz.
    for q in range(circ.num_qubits):
        tht = thetas1q[q]
        zyz = np.dot(
            np.dot(make_rz(tht[0], mat2[0]), make_ry(tht[1], mat2[1])),
            make_rz(tht[2], mat2[0]),
        )
        gate2x2_mul_mat(q, zyz, mat, workspace)

    # Multiply vector by the 2-qubit unit-block:
    #   ((Rz @ Ry) kron (Rs @ Ry)) @ entangling_gate),
    # where Rs is a swappable gate: Rx if CX, Rz if CP or CZ.
    for i in range(circ.num_blocks):
        ctrl = blocks[0, i]  # control qubit
        targ = blocks[1, i]  # target qubit

        tht = thetas2q[i]
        ctrl_mat = np.dot(make_rz(tht[1], mat2[0]), make_ry(tht[0], mat2[1]))
        targ_mat = np.dot(make_rs(tht[3], mat2[0]), make_ry(tht[2], mat2[1]))

        mat = cg_mul_mat(ctrl, targ, _get_angle(tht), mat, workspace)
        mat = gate2x2_mul_mat(ctrl, ctrl_mat, mat, workspace)
        mat = gate2x2_mul_mat(targ, targ_mat, mat, workspace)

    return mat


def v_dagger_mul_mat(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    mat: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Multiplies conjugate-transposed circuit matrix by the right-hand side matrix:
    ``mat <- V.H @ mat``. **Note**, the input matrix will be modified in-place.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of ansatz parametric circuit.
        mat: right-hand side matrix.
        workspace: workspace of size greater or equal to the size of ``mat``.

    Returns:
        modified input matrix ``mat``.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
    assert chk.complex_2d(mat, mat.shape[0] >= mat.shape[1])
    assert chk.complex_array(workspace, workspace.size >= mat.size)
    assert not np.may_share_memory(mat, workspace)
    assert mat.data.contiguous  # !!!

    blocks = circ.blocks
    mat2 = np.zeros((2, 2, 2), dtype=np.cfloat)  # temporary 2x2 matrices

    # Convenient split into parameter sub-sets for 1- and 2-qubit gates.
    thetas1q = circ.subset1q(thetas)
    thetas2q = circ.subset2q(thetas)

    # Entangling gates are: CX, CZ or CP.
    if circ.entangler == "cp":
        cg_mul_mat = cp_mul_mat  # function: controlled-gate times matrix
        make_rs = make_rz  # swappable with entangling gate

        def _get_angle(_th: np.ndarray) -> float:
            return float(-_th[4])  # mind minus sign - conjugate-transposed

    elif circ.entangler == "cz":
        cg_mul_mat = cz_mul_mat  # function: controlled-gate times matrix
        make_rs = make_rz  # swappable with entangling gate

        def _get_angle(__: np.ndarray) -> float:
            return float(0)

    else:  # cx
        cg_mul_mat = cx_mul_mat  # function: controlled-gate times matrix
        make_rs = make_rx  # swappable with entangling gate

        def _get_angle(__: np.ndarray) -> float:
            return float(0)

    # Multiply vector by the 2-qubit conjugate-transposed unit blocks:
    #   (((Rz @ Ry) kron (Rs @ Ry)) @ entangling_gate).H =
    #       entangling_gate.H @ ((Ry.H @ Rz.H) kron (Ry.H @ Rs.H)),
    # where Rs is a swappable gate: Rx if CX, Rz if CP or CZ.
    for i in range(circ.num_blocks - 1, -1, -1):
        ctrl = blocks[0, i]  # control qubit
        targ = blocks[1, i]  # target qubit

        tht = thetas2q[i]
        ctrl_mat = np.dot(make_ry(-tht[0], mat2[0]), make_rz(-tht[1], mat2[1]))
        targ_mat = np.dot(make_ry(-tht[2], mat2[0]), make_rs(-tht[3], mat2[1]))

        mat = gate2x2_mul_mat(ctrl, ctrl_mat, mat, workspace)
        mat = gate2x2_mul_mat(targ, targ_mat, mat, workspace)
        mat = cg_mul_mat(ctrl, targ, _get_angle(tht), mat, workspace)

    # Multiply vector by the initial layer of rotation gates Rz @ Ry @ Rz.
    for q in range(circ.num_qubits):
        tht = thetas1q[q]
        zyz = np.dot(
            np.dot(make_rz(-tht[2], mat2[0]), make_ry(-tht[1], mat2[1])),
            make_rz(-tht[0], mat2[0]),
        )
        gate2x2_mul_mat(q, zyz, mat, workspace)

    return mat


def grad_of_matrix_dot_product(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    x_mat: np.ndarray,
    vh_y_mat: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Computes the partial or full gradient of ``<V @ x, y> = <x, V.H @ y>``,
    where ``V = V(thetas)`` is the matrix of approximating circuit, and
    ``x``, ``y`` are matrices.

    **Note**: It is assumed that initially (!) ``vh_y_mat = V.H @ y``,
    which is a byproduct of computation of the objective function prior
    to calling this one. We do so because Qiskit separates computation of
    objective function and its gradient. By reusing ``V.H @ y`` obtained after
    objective function (it should be cached somewhere), we save a fair amount
    of computations.

    **Beware**, the input arrays ``x_mat`` and ``vh_y_mat`` will be modified
    upon completion of this function.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of the circuit.
        x_mat: left-hand side matrix; this matrix will be (!) modified.
        vh_y_mat: right-hand side matrix, expects vh_y_mat = ``V.H @ y``,
                  **not** a ``y`` itself; this matrix will be (!) modified.
        workspace: workspace of size greater or equal to the size of ``x_mat``.

    Returns:
        gradient of <V @ x, y> by theta parameters.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
    assert chk.complex_2d(x_mat)
    assert chk.complex_2d(vh_y_mat, vh_y_mat.shape == x_mat.shape)
    assert chk.complex_array(workspace, workspace.size >= x_mat.size)
    assert chk.no_overlap(x_mat, workspace)
    assert chk.no_overlap(x_mat, vh_y_mat)
    assert chk.no_overlap(vh_y_mat, workspace)
    assert x_mat.data.contiguous and vh_y_mat.data.contiguous  # !!!

    worksp = workspace.ravel()[0 : x_mat.size].reshape(x_mat.shape)
    w_mat = x_mat  # matrix w, initially w = x
    z_mat = vh_y_mat  # matrix z, initially z = V.H @ y

    # Convenient split into parameter sub-sets for 1- and 2-qubit gates.
    grad = np.zeros(circ.num_thetas, dtype=np.cfloat)
    thetas1q, grad1q = circ.subset1q(thetas), circ.subset1q(grad)
    thetas2q, grad2q = circ.subset2q(thetas), circ.subset2q(grad)

    # Entangling gates are: CX, CZ or CP. Swappable gates are: Rx (X) or Rz (Z).
    cp_entangler = False
    if circ.entangler == "cp":
        cp_entangler = True
        entangler_mul_mat = cp_mul_mat
        rs_mul_mat = rz_mul_mat
        s_dot_mat = z_dot_mat
    elif circ.entangler == "cz":
        entangler_mul_mat = cz_mul_mat
        rs_mul_mat = rz_mul_mat
        s_dot_mat = z_dot_mat
    else:  # cx
        entangler_mul_mat = cx_mul_mat
        rs_mul_mat = rx_mul_mat
        s_dot_mat = x_dot_mat

    # Compute derivatives of the front layer of 1-qubit gates.
    for q in range(circ.num_qubits):
        tht = thetas1q[q]

        grad1q[q, 2] = z_dot_mat(q, w_mat, z_mat, worksp)  # 0.5j * <Z@w|z>
        rz_mul_mat(tht[2], q, w_mat)  # w = Rz @ w
        rz_mul_mat(tht[2], q, z_mat)  # z = Rz @ z

        grad1q[q, 1] = y_dot_mat(q, w_mat, z_mat, worksp)  # 0.5j * <Y@w|z>
        ry_mul_mat(tht[1], q, w_mat, worksp)  # w = Ry @ w
        ry_mul_mat(tht[1], q, z_mat, worksp)  # z = Ry @ z

        grad1q[q, 0] = z_dot_mat(q, w_mat, z_mat, worksp)  # 0.5j * <Z@w|z>
        rz_mul_mat(tht[0], q, w_mat)  # w = Rz @ w
        rz_mul_mat(tht[0], q, z_mat)  # z = Rz @ z

    # Compute derivatives of the 2-qubit blocks.
    blocks = circ.blocks
    for i in range(circ.num_blocks):
        ctrl = blocks[0, i]  # control qubit
        targ = blocks[1, i]  # target qubit
        tht = thetas2q[i]

        # Apply the entangling 2-qubit gate and compute its derivative.
        angle = float(0)
        if cp_entangler:  # CPhase has a parameter, so a bit more complicated
            angle = float(tht[4])
            grad2q[i, 4] = derv_cphase(ctrl, targ, w_mat, z_mat, worksp)
        entangler_mul_mat(ctrl, targ, angle, z_mat, worksp)  # z = E @ z
        entangler_mul_mat(ctrl, targ, angle, w_mat, worksp)  # w = E @ w

        # Apply 1-qubit gates to control qubit.
        grad2q[i, 0] = y_dot_mat(ctrl, w_mat, z_mat, worksp)  # 0.5j * <Y@w|z>
        ry_mul_mat(tht[0], ctrl, w_mat, worksp)  # w = Ry @ w
        ry_mul_mat(tht[0], ctrl, z_mat, worksp)  # z = Ry @ z

        grad2q[i, 1] = z_dot_mat(ctrl, w_mat, z_mat, worksp)  # 0.5j * <Z@w|z>
        rz_mul_mat(tht[1], ctrl, w_mat)  # w = Rz @ w
        rz_mul_mat(tht[1], ctrl, z_mat)  # z = Rz @ z

        # Apply 1-qubit gates to target qubit. Mind the Swappable gate.
        grad2q[i, 2] = y_dot_mat(targ, w_mat, z_mat, worksp)  # 0.5j * <Y@w|z>
        ry_mul_mat(tht[2], targ, w_mat, worksp)  # w = Ry @ w
        ry_mul_mat(tht[2], targ, z_mat, worksp)  # z = Ry @ z

        grad2q[i, 3] = s_dot_mat(targ, w_mat, z_mat, worksp)  # 0.5j * <S@w|z>
        rs_mul_mat(tht[3], targ, w_mat, worksp)  # w = Rs @ w
        rs_mul_mat(tht[3], targ, z_mat, worksp)  # z = Rs @ z

    return grad


def coord_descent_single_sweep(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    target: np.ndarray,
    workspace: np.ndarray,
) -> float:
    """
    Sweeps over all the angular parameters and optimizes the objective function
    by coordinate descent. Here we consider the following objective:
    ``fobj = 1 - |<V,U>|^2 / dim^2``. If the second derivative of ``fobj``
    by a parameter is strictly positive, Newton iteration is made, otherwise
    a simple gradient descent. That is, just one updating iteration is carried
    out for every parameter such that every entry of the array ``thetas`` will
    be modified by the end. The actual optimization implies that this function
    is invoked many times until the change of parameters becomes negligible.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of the circuit.
        target: target unitary matrix to be approximated.
        workspace: workspace of size greater or equal to the triple size of ``target``.

    Returns:
        the value of objective function given thetas at the end of a sweep;
        parameters ``thetas`` will be modified *in-place*.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
    assert chk.complex_2d_square(target)
    assert chk.complex_array(workspace, workspace.size >= 3 * target.size)
    assert chk.no_overlap(target, workspace)

    tol = float(np.sqrt(np.finfo(np.float64).eps))  # tolerance for positiveness
    dim = target.shape[0]

    # Allocate temporary arrays.
    workspace = workspace.ravel()[0 : 3 * target.size].reshape((3, dim, dim))
    w_mat = workspace[0]  # matrix w
    z_mat = workspace[1]  # matrix z
    worksp = workspace[2]

    # Initialize the temporary array, w := x = I, z := V.H @ y = V.H @ U @ x.
    w_mat.fill(0)
    np.fill_diagonal(w_mat, 1)  # w_mat := x = eye(dim) = I
    np.copyto(z_mat, target)  # z_mat := y = U @ x = U @ I = U
    v_dagger_mul_mat(circ, thetas, z_mat, worksp)  # z_mat = V.H @ y

    # Convenient split into parameter sub-sets for 1- and 2-qubit gates.
    thetas1q = circ.subset1q(thetas)
    thetas2q = circ.subset2q(thetas)

    # Entangling gates are: CX, CZ or CP. Swappable gates are: Rx (X) or Rz (Z).
    cp_entangler = False
    if circ.entangler == "cp":  # pylint: disable=no-else-raise
        raise NotImplementedError("CPhase entangler is not supported yet")
    elif circ.entangler == "cz":
        entangler_mul_mat = cz_mul_mat
        rs_mul_mat = rz_mul_mat
        s_dot_mat = z_dot_mat
    else:  # cx
        entangler_mul_mat = cx_mul_mat
        rs_mul_mat = rx_mul_mat
        s_dot_mat = x_dot_mat

    # Define function that computes the angle increment.
    learn_rate = np.pi / 16
    max_delta_theta = np.pi / 4

    def _delta_theta(prod_: np.cfloat, grad_: np.cfloat) -> float:
        """
        Computes theta increment from the product <V,U> and its gradient.
        f_obj = 1 - |<V,U>|^2 / dim^2
        """
        # First and second derivatives of objective function:
        derv1_ = (-2.0 * np.real(np.conj(prod_) * grad_)) / (dim**2)
        derv2_ = (-2.0 * abs(grad_) ** 2 + 0.5 * abs(prod_) ** 2) / (dim**2)

        # Newton iteration, if derv2 is strictly positive, otherwise grad. descent.
        if derv2_ < tol:
            derv1_ /= max(abs(derv1_), 1.0)  # |fobj_grad| <= 1
            dt_ = -learn_rate * derv1_
        else:
            dt_ = -derv1_ / derv2_

        abs_dt_ = abs(dt_ / max_delta_theta)
        return dt_ if abs_dt_ <= 1 else dt_ / abs_dt_  # |dt| <= max_delta_theta

    # Compute derivatives of the front layer of 1-qubit gates.
    for q in range(circ.num_qubits):
        tht = thetas1q[q]

        grad = z_dot_mat(q, w_mat, z_mat, worksp)  # 0.5j * <Z@w|z>
        prod = np.vdot(w_mat, z_mat)
        rz_mul_mat(tht[2], q, z_mat)  # z = Rz @ z
        tht[2] += _delta_theta(prod, grad)
        rz_mul_mat(tht[2], q, w_mat)  # w = Rz @ w

        grad = y_dot_mat(q, w_mat, z_mat, worksp)  # 0.5j * <Y@w|z>
        prod = np.vdot(w_mat, z_mat)
        ry_mul_mat(tht[1], q, z_mat, worksp)  # z = Ry @ z
        tht[1] += _delta_theta(prod, grad)
        ry_mul_mat(tht[1], q, w_mat, worksp)  # w = Ry @ w

        grad = z_dot_mat(q, w_mat, z_mat, worksp)  # 0.5j * <Z@w|z>
        prod = np.vdot(w_mat, z_mat)
        rz_mul_mat(tht[0], q, z_mat)  # z = Rz @ z
        tht[0] += _delta_theta(prod, grad)
        rz_mul_mat(tht[0], q, w_mat)  # w = Rz @ w

    # Compute derivatives of the 2-qubit blocks.
    blocks = circ.blocks
    for i in range(circ.num_blocks):
        ctrl = blocks[0, i]  # control qubit
        targ = blocks[1, i]  # target qubit
        tht = thetas2q[i]

        # Apply the entangling 2-qubit gate and compute its derivative.
        angle = float(0)
        if cp_entangler:  # CPhase has a parameter, so a bit more complicated
            raise NotImplementedError("CPhase entangler is not supported yet")
        entangler_mul_mat(ctrl, targ, angle, z_mat, worksp)  # z = E @ z
        entangler_mul_mat(ctrl, targ, angle, w_mat, worksp)  # w = E @ w

        # Apply 1-qubit gates to control qubit.
        grad = y_dot_mat(ctrl, w_mat, z_mat, worksp)  # 0.5j * <Y@w|z>
        prod = np.vdot(w_mat, z_mat)
        ry_mul_mat(tht[0], ctrl, z_mat, worksp)  # z = Ry @ z
        tht[0] += _delta_theta(prod, grad)
        ry_mul_mat(tht[0], ctrl, w_mat, worksp)  # w = Ry @ w

        grad = z_dot_mat(ctrl, w_mat, z_mat, worksp)  # 0.5j * <Z@w|z>
        prod = np.vdot(w_mat, z_mat)
        rz_mul_mat(tht[1], ctrl, z_mat)  # z = Rz @ z
        tht[1] += _delta_theta(prod, grad)
        rz_mul_mat(tht[1], ctrl, w_mat)  # w = Rz @ w

        # Apply 1-qubit gates to target qubit. Mind the Swappable gate.
        grad = y_dot_mat(targ, w_mat, z_mat, worksp)  # 0.5j * <Y@w|z>
        prod = np.vdot(w_mat, z_mat)
        ry_mul_mat(tht[2], targ, z_mat, worksp)  # z = Ry @ z
        tht[2] += _delta_theta(prod, grad)
        ry_mul_mat(tht[2], targ, w_mat, worksp)  # w = Ry @ w

        grad = s_dot_mat(targ, w_mat, z_mat, worksp)  # 0.5j * <S@w|z>
        prod = np.vdot(w_mat, z_mat)
        rs_mul_mat(tht[3], targ, z_mat, worksp)  # z = Rs @ z
        tht[3] += _delta_theta(prod, grad)
        rs_mul_mat(tht[3], targ, w_mat, worksp)  # w = Rs @ w

    # Compute the objective function at the end. Ideally, we should recompute
    # <x|V.H@y> from scratch with updated thetas, however, the code below is
    # more economical albeit it accumulates round-off errors.
    return float(1 - np.abs(np.vdot(w_mat, z_mat) / dim) ** 2)
