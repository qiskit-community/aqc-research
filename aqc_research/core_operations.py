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
Core operations for an approximating quantum compiling.
We exploit sparsity patterns of the gate matrices avoiding their explicit
construction (as full size ``2^n x 2^n`` matrices) for memory and
performance efficiency.
"""

from typing import Optional, Tuple
import numpy as np
import aqc_research.utils as helper
import aqc_research.checking as chk
from aqc_research.parametric_circuit import ParametricCircuit, TrotterAnsatz
from aqc_research.elementary_operations import np_phase, np_x, np_z, make_rx, make_ry, make_rz

_logger = helper.create_logger(__file__)

# Projectors |0><0| and |1><1| respectively:
_glo_mat00 = np.asarray([[1, 0], [0, 0]], dtype=np.cfloat)
_glo_mat11 = np.asarray([[0, 0], [0, 1]], dtype=np.cfloat)


def bit2bit_transform(n: int, i: int) -> int:
    """
    Transforms bit ordering. **Note**, in Numpy implementation we flip
    bit ordering to conform to Qiskit convention.

    Args:
        n: number of qubits.
        i: qubit position.
    """
    return n - 1 - i


def gate2x2_mul_vec(
    num_qubits: int,
    pos: int,
    gate2x2: np.ndarray,
    vec: np.ndarray,
    out: np.ndarray,
    inplace: bool,
) -> np.ndarray:
    """
    Computes the product of 2x2 gate matrix times right-hand size vector.
    Args:
        num_qubits: number of qubits, ``n >= 2``.
        pos: position of a qubit where 2x2 gate is applied.
        gate2x2: ``2x2`` gate matrix.
        vec: right-hand side vector to be multiplied.
        out: placeholder for output result or a workspace of the same size as ``vec``.
        inplace: if True, the result of multiplication will be formed in ``vec``
                 itself and the placeholder ``out`` will be used as a workspace,
                 otherwise, the result will be placed into ``out``.
    Returns:
        depending on the flag ``inplace`` either ``vec`` or ``out``, where
        the result of multiplication is formed.
    """
    assert 0 <= pos < num_qubits
    assert gate2x2.shape == (2, 2)
    assert vec.shape == out.shape == (2**num_qubits,)
    assert vec.data.contiguous and out.data.contiguous
    assert not np.may_share_memory(vec, out)
    assert gate2x2.dtype == vec.dtype == out.dtype
    assert isinstance(inplace, bool)

    h = 2 ** (num_qubits - pos - 1)
    g00, g01, g10, g11 = gate2x2[0, 0], gate2x2[0, 1], gate2x2[1, 0], gate2x2[1, 1]

    out[:] = vec[:]
    ro = vec.reshape((-1, 2 * h)) if inplace else out.reshape((-1, 2 * h))
    rv = out.reshape((-1, 2 * h)) if inplace else vec.reshape((-1, 2 * h))

    # Compute top part of 2x2 gate matrix times vector.
    if g00 != 0:
        ro[:, :h] *= g00
    else:
        ro[:, :h] = 0

    if g01 != 0:
        ro[:, h:] *= g01
    else:
        ro[:, h:] = 0

    ro[:, :h] += ro[:, h:]

    # Compute bottom part of 2x2 gate matrix times vector.
    if abs(g10) > abs(g11):
        if g11 != 0:
            ro[:, h:] = rv[:, h:]
            ro[:, h:] *= g11 / g10
        else:
            ro[:, h:] = 0

        ro[:, h:] += rv[:, :h]
        ro[:, h:] *= g10
    elif abs(g11) > 0:
        if g10 != 0:
            ro[:, h:] = rv[:, :h]
            ro[:, h:] *= g10 / g11
        else:
            ro[:, h:] = 0

        ro[:, h:] += rv[:, h:]
        ro[:, h:] *= g11
    else:
        ro[:, h:] = 0

    return vec if inplace else out


def proj00_mul_vec(num_qubits: int, pos: int, vec: np.ndarray) -> np.ndarray:
    """
    Computes the product of a layer consisting of a single 1-qubit projector
    |0><0| times right-hand size vector.
    **Note**, the vector will be modified in-place.
    Args:
        num_qubits: number of qubits, ``n >= 2``.
        pos: position of a qubit where 2x2 gate is applied.
        vec: right-hand side vector to be multiplied in-place.
    Returns:
        input vector modified in-place.
    """
    assert 0 <= pos < num_qubits
    assert vec.shape == (2**num_qubits,) and vec.data.contiguous

    h = 2 ** (num_qubits - pos - 1)
    v = vec.reshape((-1, 2 * h))
    v[:, h:] = 0
    return vec


def proj11_mul_vec(num_qubits: int, pos: int, vec: np.ndarray) -> np.ndarray:
    """
    Computes the product of a layer consisting of a single 1-qubit projector
    |1><1| times right-hand size vector.
    **Note**, the vector will be modified in-place.
    Args:
        num_qubits: number of qubits, ``n >= 2``.
        pos: position of a qubit where 2x2 gate is applied.
        vec: right-hand side vector to be multiplied in-place.
    Returns:
        input vector modified in-place.
    """
    assert 0 <= pos < num_qubits
    assert vec.shape == (2**num_qubits,) and vec.data.contiguous

    h = 2 ** (num_qubits - pos - 1)
    v = vec.reshape((-1, 2 * h))
    v[:, :h] = 0
    return vec


def rx_mul_vec(
    n: int,
    pos: int,
    angle: float,
    vec: np.ndarray,
    temp: np.ndarray,
) -> np.ndarray:
    """
    Computes the product of Rx-gate matrix times right-hand side vector.
    **Note**, the vector will be modified in-place.

    Args:
        n: number of qubits, ``n >= 2``.
        pos: position of a qubit where 2x2 gate is applied.
        angle: angular parameter of the gate.
        vec: right-hand side vector to be multiplied in-place.
        temp: temporary array of shape == vec.shape.

    Returns:
        input vector modified in-place.
    """
    assert 0 <= pos < n and chk.is_float(angle)
    assert vec.shape == temp.shape == (2**n,)
    assert vec.data.contiguous and temp.data.contiguous
    assert not np.may_share_memory(vec, temp)

    h = 2 ** (n - pos - 1)
    np.multiply(vec, -1j * np.sin(0.5 * angle), out=temp)
    vec *= np.cos(0.5 * angle)
    v = vec.reshape((-1, 2 * h))
    w = temp.reshape((-1, 2 * h))
    v[:, :h] += w[:, h:]
    v[:, h:] += w[:, :h]
    return vec


def ry_mul_vec(
    n: int,
    pos: int,
    angle: float,
    vec: np.ndarray,
    temp: np.ndarray,
) -> np.ndarray:
    """
    Computes the product of Ry-gate matrix times right-hand side vector.
    **Note**, the vector will be modified in-place.

    Args:
        n: number of qubits, ``n >= 2``.
        pos: position of a qubit where 2x2 gate is applied.
        angle: angular parameter of the gate.
        vec: right-hand side vector to be multiplied in-place.
        temp: temporary array of shape == vec.shape.

    Returns:
        input vector modified in-place.
    """
    assert 0 <= pos < n and chk.is_float(angle)
    assert vec.shape == temp.shape == (2**n,)
    assert vec.data.contiguous and temp.data.contiguous
    assert not np.may_share_memory(vec, temp)

    h = 2 ** (n - pos - 1)
    np.multiply(vec, np.sin(0.5 * angle), out=temp)
    vec *= np.cos(0.5 * angle)
    v = vec.reshape((-1, 2 * h))
    w = temp.reshape((-1, 2 * h))
    v[:, :h] -= w[:, h:]
    v[:, h:] += w[:, :h]
    return vec


def rz_mul_vec(
    n: int,
    pos: int,
    angle: float,
    vec: np.ndarray,
    _: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes the product of Rz-gate matrix times right-hand side vector.
    **Note**, the vector will be modified in-place.

    Args:
        n: number of qubits, ``n >= 2``.
        pos: position of a qubit where 2x2 gate is applied.
        angle: angular parameter of the gate.
        vec: right-hand side vector to be multiplied in-place.
        _: argument retained for interface compatibility.

    Returns:
        input vector modified in-place.
    """
    assert 0 <= pos < n and chk.is_float(angle)
    assert vec.shape == (2**n,) and vec.data.contiguous

    h = 2 ** (n - pos - 1)
    v = vec.reshape((-1, 2 * h))
    v[:, :h] *= np.exp(-0.5j * angle)
    v[:, h:] *= np.exp(+0.5j * angle)
    return vec


def dot_x(
    n: int, pos: int, w_vec: np.ndarray, z_vec: np.ndarray, temp: np.ndarray
) -> np.complex128:
    """
    Computes 0.5j * <X@w|z>.
    Important: vdot() is the most effective on contiguous arrays. This is why
    we first form a vector |Pauli_gate @ w_vec> and then apply vdot(). Although,
    a temporary vector could be avoided in principle, this approach is faster.
    Args:
        n: number of qubits, ``n >= 2``.
        pos: position of a qubit where 2x2 gate is applied.
        w_vec: w vector.
        z_vec: y vector.
        temp: temporary vector for intermediate results.
    Returns:
        0.5j * <X@w|z>.
    """
    assert 0 <= pos < n
    assert w_vec.shape == z_vec.shape == temp.shape == (2**n,)
    assert w_vec.data.contiguous and z_vec.data.contiguous and temp.data.contiguous

    h = 2 ** (n - pos - 1)
    iv = w_vec.reshape((-1, 2 * h))
    ov = temp.reshape((-1, 2 * h))
    ov[:, :h] = iv[:, h:]
    ov[:, h:] = iv[:, :h]
    return 0.5j * np.vdot(temp, z_vec)


def dot_y(
    n: int, pos: int, w_vec: np.ndarray, z_vec: np.ndarray, temp: np.ndarray
) -> np.complex128:
    """
    Computes 0.5j * <Y@w|z>.
    Important: vdot() is the most effective on contiguous arrays. This is why
    we first form a vector |Pauli_gate @ w_vec> and then apply vdot(). Although,
    a temporary vector could be avoided in principle, this approach is faster.
    Args:
        n: number of qubits, ``n >= 2``.
        pos: position of a qubit where 2x2 gate is applied.
        w_vec: w vector.
        z_vec: y vector.
        temp: temporary vector for intermediate results.
    Returns:
        0.5j * <Y@w|z>.
    """
    assert 0 <= pos < n
    assert w_vec.shape == z_vec.shape == temp.shape == (2**n,)
    assert w_vec.data.contiguous and z_vec.data.contiguous and temp.data.contiguous

    h = 2 ** (n - pos - 1)
    iv = w_vec.reshape((-1, 2 * h))
    ov = temp.reshape((-1, 2 * h))
    np.negative(iv[:, h:], out=ov[:, :h])
    ov[:, h:] = iv[:, :h]
    return 0.5 * np.vdot(temp, z_vec)  # 0.5 = 0.5j * conj(j)


def dot_z(
    n: int, pos: int, w_vec: np.ndarray, z_vec: np.ndarray, temp: np.ndarray
) -> np.complex128:
    """
    Computes 0.5j * <Z@w|z>.
    Important: vdot() is the most effective on contiguous arrays. This is why
    we first form a vector |Pauli_gate @ w_vec> and then apply vdot(). Although,
    a temporary vector could be avoided in principle, this approach is faster.
    Args:
        n: number of qubits, ``n >= 2``.
        pos: position of a qubit where 2x2 gate is applied.
        w_vec: w vector.
        z_vec: y vector.
        temp: temporary vector for intermediate results.
    Returns:
        0.5j * <Z@w|z>.
    """
    assert 0 <= pos < n
    assert w_vec.shape == z_vec.shape == temp.shape == (2**n,)
    assert w_vec.data.contiguous and z_vec.data.contiguous and temp.data.contiguous

    h = 2 ** (n - pos - 1)
    iv = w_vec.reshape((-1, 2 * h))
    ov = temp.reshape((-1, 2 * h))
    ov[:] = iv[:]  # faster than just:  ov[:, :h] = iv[:, :h]
    np.negative(iv[:, h:], out=ov[:, h:])
    return 0.5j * np.vdot(temp, z_vec)


def block_mul_vec(
    n: int,
    c: int,
    t: int,
    c_mat: np.ndarray,
    t_mat: np.ndarray,
    g_mat: np.ndarray,
    vec: np.ndarray,
    workspace: np.ndarray,
    dagger: bool,
) -> np.ndarray:
    """
    Multiplies in-place the input vector by the block matrix. The block
    is assumed having the following structure:
                            _______
        control -----*-----| c_mat |-----
                     |      -------
                     |      _______
        target  ----|G|----| t_mat |-----
                            -------

    **Note**, if ``dagger=True`` we do **not** conjugate-transpose the input
    matrices, assuming that user did so already. We just flip the block
    structure horizontally.

    Args:
        n: number of qubits.
        c: index of control qubit.
        t: index of target qubit.
        c_mat: 1-qubit 2x2 matrix applied at control qubit.
        t_mat: 1-qubit 2x2 matrix applied at target qubit.
        g_mat: 1-qubit 2x2 matrix of controlled gate (G) of entangler.
        vec: vector to be multiplied by the block matrix (!) in-place.
        workspace: temporary array of shape (2, vec.size).
        dagger: if True, the above 2-qubit block structure should be flipped.

    Returns:
        input vector multiplied by block matrix in-place: vec = block_mat @ vec.
    """
    assert 0 <= c < n and 0 <= t < n and c != t
    assert c_mat.shape == t_mat.shape == g_mat.shape == (2, 2)
    assert vec.shape == (2**n,) and workspace.shape == (2, vec.size)
    assert vec.data.contiguous and workspace.data.contiguous
    assert not np.may_share_memory(vec, workspace)
    assert c_mat.dtype == t_mat.dtype == g_mat.dtype == vec.dtype == workspace.dtype

    if dagger:
        c00 = _glo_mat00 @ c_mat  # |0><0| @ control matrix
        c11 = _glo_mat11 @ c_mat  # |1><1| @ control matrix
        t_g = g_mat @ t_mat  # gate matrix @ target matrix
    else:
        c00 = c_mat @ _glo_mat00  # control matrix @ |0><0|
        c11 = c_mat @ _glo_mat11  # control matrix @ |1><1|
        t_g = t_mat @ g_mat  # target matrix @ gate matrix

    # (c_mat@|0><0| or |0><0|@c_mat) kron (t_mat@ident) --> workspace[0]:
    gate2x2_mul_vec(n, c, c00, vec, workspace[1], False)
    gate2x2_mul_vec(n, t, t_mat, workspace[1], workspace[0], False)

    # (c_mat@|1><1| or |1><1|@c_mat) kron (t_mat@g_mat or g_mat@t_mat) --> vec:
    gate2x2_mul_vec(n, c, c11, vec, workspace[1], False)
    gate2x2_mul_vec(n, t, t_g, workspace[1], vec, False)

    # Sum up two intermediate results and return.
    vec += workspace[0]
    return vec


def cx_mul_vec(
    n: int,
    c: int,
    t: int,
    _: float,
    vec: np.ndarray,
    temp: np.ndarray,
) -> np.ndarray:
    """
    Computes the product of a layer consisting of single 2-qubit operator that
    represents CNot gate times right-hand size vector.
    **Note**, the vector will be modified in-place.
    Args:
        n: number of qubits.
        c: index of control qubit.
        t: index of target qubit.
        _: argument retained for interface compatibility.
        vec: right-hand side vector to be multiplied in-place.
        temp: temporary array of shape == vec.shape.
    Returns:
        input vector modified in-place.
    """
    assert 0 <= c < n and 0 <= t < n and c != t
    assert vec.shape == temp.shape == (2**n,)
    assert vec.data.contiguous and temp.data.contiguous
    assert not np.may_share_memory(vec, temp)

    # temp = full_matrix(|1><1|_{c}) @ vec.
    h = 2 ** (n - c - 1)
    w = temp.reshape((-1, 2 * h))
    v = vec.reshape((-1, 2 * h))
    w[:, :h] = 0
    w[:, h:] = v[:, h:]

    # vec = full_matrix(|0><0|_{c} @ I_{t}) @ vec.
    v[:, h:] = 0

    # vec = vec + full_matrix(|1><1|_{c} @ X_{t}) @ vec.
    h = 2 ** (n - t - 1)
    w = temp.reshape((-1, 2 * h))
    v = vec.reshape((-1, 2 * h))
    v[:, :h] += w[:, h:]
    v[:, h:] += w[:, :h]
    return vec


def cz_mul_vec(
    n: int,
    c: int,
    t: int,
    _: float,
    vec: np.ndarray,
    temp: np.ndarray,
) -> np.ndarray:
    """
    Computes the product of a layer consisting of single 2-qubit operator that
    represents CZ gate times right-hand size vector.
    **Note**, the vector will be modified in-place.
    Args:
        n: number of qubits.
        c: index of control qubit.
        t: index of target qubit.
        _: argument retained for interface compatibility.
        vec: right-hand side vector to be multiplied in-place.
        temp: temporary array of shape == vec.shape.
    Returns:
        input vector modified in-place.
    """
    assert 0 <= c < n and 0 <= t < n and c != t
    assert vec.shape == temp.shape == (2**n,)
    assert vec.data.contiguous and temp.data.contiguous
    assert not np.may_share_memory(vec, temp)

    # temp = full_matrix(|1><1|_{c}) @ vec.
    h = 2 ** (n - c - 1)
    w = temp.reshape((-1, 2 * h))
    v = vec.reshape((-1, 2 * h))
    w[:, :h] = 0
    w[:, h:] = v[:, h:]

    # vec = full_matrix(|0><0|_{c} @ I_{t}) @ vec.
    v[:, h:] = 0

    # vec = vec + full_matrix(|1><1|_{c} @ Z_{t}) @ vec.
    h = 2 ** (n - t - 1)
    w = temp.reshape((-1, 2 * h))
    v = vec.reshape((-1, 2 * h))
    v[:, :h] += w[:, :h]
    v[:, h:] -= w[:, h:]
    return vec


def cp_mul_vec(
    n: int,
    c: int,
    t: int,
    angle: float,
    vec: np.ndarray,
    temp: np.ndarray,
) -> np.ndarray:
    """
    Computes the product of a layer consisting of single 2-qubit operator that
    represents CPhase gate times right-hand size vector.
    **Note**, the vector will be modified in-place.

    Args:
        n: number of qubits.
        c: index of control qubit.
        t: index of target qubit.
        angle: argument retained for interface compatibility.
        vec: right-hand side vector to be multiplied in-place.
        temp: temporary array of shape == vec.shape.

    Returns:
        input vector modified in-place.
    """
    assert 0 <= c < n and 0 <= t < n and c != t and chk.is_float(angle)
    assert vec.shape == temp.shape == (2**n,)
    assert vec.data.contiguous and temp.data.contiguous
    assert not np.may_share_memory(vec, temp)

    # temp = full_matrix(|1><1|_{c}) @ vec.
    h = 2 ** (n - c - 1)
    w = temp.reshape((-1, 2 * h))
    v = vec.reshape((-1, 2 * h))
    w[:, :h] = 0
    w[:, h:] = v[:, h:]

    # vec = full_matrix(|0><0|_{c} @ I_{t}) @ vec.
    v[:, h:] = 0

    # vec = vec + full_matrix(|1><1|_{c} @ P_{t}) @ vec.
    h = 2 ** (n - t - 1)
    w = temp.reshape((-1, 2 * h))
    np.multiply(w[:, h:], np.exp(1j * angle), out=w[:, h:])
    vec += temp
    return vec


def derv_cphase_mul_vec(
    n: int,
    c: int,
    t: int,
    angle: float,
    vec: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """
    Computes the product of a layer consisting of single 2-qubit operator that
    represents a derivative of CPhase gate times right-hand size vector.
    Actually, the function can compute the result in-place, however, the way
    we use it in the code requires separate ``vec`` and ``out`` arrays.

    Args:
        n: number of qubits.
        c: index of control qubit.
        t: index of target qubit.
        angle: angular parameter of the phase gate.
        vec: right-hand side vector.
        out: placeholder for output result; must be different from ``vec``.

    Returns:
        output vector filled up by multiplication result.
    """
    assert 0 <= c < n and 0 <= t < n and c != t and chk.is_float(angle)
    assert vec.shape == out.shape == (2**n,) and vec.dtype == out.dtype
    assert vec.data.contiguous and out.data.contiguous
    assert not np.may_share_memory(vec, out)

    # vec = full_matrix(|1><1|_{c}) @ vec.
    h = 2 ** (n - c - 1)
    iv = vec.reshape((-1, 2 * h))
    ov = out.reshape((-1, 2 * h))
    ov[:, :h] = 0
    ov[:, h:] = iv[:, h:]

    # vec = full_matrix(derivative_of_phase_gate_{t}) @ vec.
    h = 2 ** (n - t - 1)
    vo = out.reshape((-1, 2 * h))
    vo[:, :h] = 0
    vo[:, h:] *= 1j * np.exp(1j * angle)
    return out


def v_mul_vec(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    vec: np.ndarray,
    out: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Multiplies circuit matrix by the right-hand side vector: ``out = V @ vec``.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of ansatz parametric circuit.
        vec: right-hand side vector.
        out: placeholder for the output result.
        workspace: array of temporary vectors of shape (>=2, vec.size).

    Returns:
        out = V @ vec.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
    assert chk.check_sim_complex_vecs4(vec, out, workspace[0], workspace[1])
    assert chk.no_overlap(vec, workspace)
    assert chk.no_overlap(out, workspace)

    b2b = bit2bit_transform  # shorthand alias
    blocks = circ.blocks
    mat2 = np.zeros((2, 2, 2), dtype=np.cfloat)  # temporary 2x2 matrices
    n = circ.num_qubits
    trotterized = isinstance(circ, TrotterAnsatz)

    # Add a half-layer at the end of the 2nd order Trotter circuit.
    last_half_layer_num_blocks = int(0)
    if isinstance(circ, TrotterAnsatz) and circ.is_second_order:
        last_half_layer_num_blocks = circ.half_layer_num_blocks

    # Initially out = vec.
    np.copyto(out, vec)

    # Convenient split into parameter sub-sets for 1- and 2-qubit gates.
    thetas1q = circ.subset1q(thetas)
    thetas2q = circ.subset2q(thetas)

    # Entangling gates are: CX, CZ or CP.
    x_entangler, z_entangler = np_x(), np_z()
    if circ.entangler == "cp":
        make_rs = make_rz  # swappable with entangling gate

        def entangler(_th: np.ndarray) -> np.ndarray:
            return np_phase(_th[4])

    elif circ.entangler == "cz":
        make_rs = make_rz  # swappable with entangling gate

        def entangler(_) -> np.ndarray:
            return z_entangler

    else:  # cx
        make_rs = make_rx  # swappable with entangling gate

        def entangler(_) -> np.ndarray:
            return x_entangler

    # Multiply vector by the initial layer of rotation gates Rz @ Ry @ Rz.
    for q in range(n):
        tht = thetas1q[q]
        zyz = np.dot(
            np.dot(make_rz(tht[0], mat2[0]), make_ry(tht[1], mat2[1])),
            make_rz(tht[2], mat2[0]),
        )
        gate2x2_mul_vec(n, b2b(n, q), zyz, out, workspace[0], True)

    # Multiply vector by the 2-qubit unit-block:
    #   ((Rz @ Ry) kron (Rs @ Ry)) @ entangling_gate),
    # where Rs is a swappable gate: Rx if CX, Rz if CP or CZ.
    #   Recall, "num_blocks" counts the number of blocks in full layers.
    # The latter does not include the blocks of the half-layer at the end,
    # in case of 2nd order Trotter circuit. Extra half-layer will be set
    # identical to the front one.
    for i in range(circ.num_blocks + last_half_layer_num_blocks):
        i_mod_nb = i % circ.num_blocks
        ctrl = b2b(n, blocks[0, i_mod_nb])  # control qubit
        targ = b2b(n, blocks[1, i_mod_nb])  # target qubit
        tht = thetas2q[i_mod_nb]

        if trotterized and i % 3 == 0:  # block triplet starts with Rz
            rz_mul_vec(n, ctrl, -np.pi / 2, out)

        block_mul_vec(
            n,
            ctrl,
            targ,
            np.dot(make_rz(tht[1], mat2[0]), make_ry(tht[0], mat2[1])),
            np.dot(make_rs(tht[3], mat2[0]), make_ry(tht[2], mat2[1])),
            entangler(tht),
            out,
            workspace[0:2],
            dagger=False,
        )

        if trotterized and i % 3 == 2:  # block triplet ends with Rz
            rz_mul_vec(n, targ, np.pi / 2, out)

    return out


def v_dagger_mul_vec(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    vec: np.ndarray,
    out: np.ndarray,
    workspace: np.ndarray,
) -> np.ndarray:
    """
    Multiplies conjugate-transposed circuit matrix by right-hand side vector:
    ``out = V.H @ vec``. Note, the gates are applied to the vector in inverse
    order because conjugate-transposed matrix is flipped over.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of ansatz parametric circuit.
        vec: right-hand side vector.
        out: placeholder for the output result.
        workspace: array of temporary vectors of shape (>=2, vec.size).

    Returns:
        out = V.H @ vec.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
    assert chk.check_sim_complex_vecs4(vec, out, workspace[0], workspace[1])
    assert chk.no_overlap(vec, workspace)
    assert chk.no_overlap(out, workspace)

    b2b = bit2bit_transform  # shorthand alias
    blocks = circ.blocks
    mat2 = np.zeros((2, 2, 2), dtype=np.cfloat)  # temporary 2x2 matrices
    n = circ.num_qubits
    trotterized = isinstance(circ, TrotterAnsatz)

    # Add a half-layer at the end of the 2nd order Trotter circuit.
    last_half_layer_num_blocks = int(0)
    if isinstance(circ, TrotterAnsatz) and circ.is_second_order:
        last_half_layer_num_blocks = circ.half_layer_num_blocks

    # Initially out = vec.
    np.copyto(out, vec)

    # Convenient split into parameter sub-sets for 1- and 2-qubit gates.
    thetas1q = circ.subset1q(thetas)
    thetas2q = circ.subset2q(thetas)

    # Entangling gates are: CX, CZ or CP.
    x_entangler, z_entangler = np_x(), np_z()
    if circ.entangler == "cp":
        make_rs = make_rz  # swappable with entangling gate

        def entangler(_th: np.ndarray) -> np.ndarray:
            return np_phase(-_th[4])  # mind minus sign - conjugate-transposed

    elif circ.entangler == "cz":
        make_rs = make_rz  # swappable with entangling gate

        def entangler(_) -> np.ndarray:
            return z_entangler

    else:  # cx
        make_rs = make_rx  # swappable with entangling gate

        def entangler(_) -> np.ndarray:
            return x_entangler

    # Multiply vector by the 2-qubit conjugate-transposed unit blocks:
    #   (((Rz @ Ry) kron (Rs @ Ry)) @ entangling_gate).H =
    #       entangling_gate.H @ ((Ry.H @ Rz.H) kron (Ry.H @ Rs.H)),
    # where Rs is a swappable gate: Rx if CX, Rz if CP or CZ.
    #   Recall, "num_blocks" counts the number of blocks in full layers.
    # The latter does not include the blocks of the half-layer at the end,
    # in case of 2nd order Trotter circuit. Extra half-layer will be set
    # identical to the front one.
    for i in range(circ.num_blocks + last_half_layer_num_blocks - 1, -1, -1):
        i_mod_nb = i % circ.num_blocks
        ctrl = b2b(n, blocks[0, i_mod_nb])  # control qubit
        targ = b2b(n, blocks[1, i_mod_nb])  # target qubit
        tht = thetas2q[i_mod_nb]

        if trotterized and i % 3 == 2:  # block triplet "ends" (mind V.H) with Rz
            rz_mul_vec(n, targ, -np.pi / 2, out)

        block_mul_vec(
            n,
            ctrl,  # control qubit
            targ,  # target qubit
            np.dot(make_ry(-tht[0], mat2[0]), make_rz(-tht[1], mat2[1])),
            np.dot(make_ry(-tht[2], mat2[0]), make_rs(-tht[3], mat2[1])),
            entangler(tht),
            out,
            workspace[0:2],
            dagger=True,
        )

        if trotterized and i % 3 == 0:  # block triplet "starts" (mind V.H) with Rz
            rz_mul_vec(n, ctrl, np.pi / 2, out)

    # Multiply vector by the initial layer of rotation gates: (Rz @ Ry @ Rz).H.
    for q in range(n):
        tht = thetas1q[q]
        zyz = np.dot(
            np.dot(make_rz(-tht[2], mat2[0]), make_ry(-tht[1], mat2[1])),
            make_rz(-tht[0], mat2[0]),
        )
        gate2x2_mul_vec(n, b2b(n, q), zyz, out, workspace[0], True)

    return out


def grad_of_dot_product(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    x_vec: np.ndarray,
    vh_y_vec: np.ndarray,
    workspace: np.ndarray,
    block_range: Optional[Tuple[int, int]] = None,
    front_layer: bool = True,
) -> np.ndarray:
    """
    Computes the partial or full gradient of ``<V @ x, y> = <x, V.H @ y>``,
    where ``V = V(thetas)`` is the matrix of approximating circuit, and
    ``x``, ``y`` are vectors.

    Gradients are computed for the front layer of 1-qubit gates, if enabled, and
    for 2-qubit unit blocks in the range ``[ block_range[0], block_range[1] )``
    (mind semi-open interval). For the blocks outside the range this function
    will return zero gradients. Default values in the last two arguments imply
    computation of the full gradient.

    **Note**: It is assumed that initially (!) ``vh_y_vec = V.H @ y``,
    which is a byproduct of computation of the objective function prior
    to calling this one. We do so because Qiskit separates computation of
    objective function and its gradient. By reusing ``V.H @ y`` obtained after
    objective function (it should be cached somewhere), we save a fair amount
    of computations.

    **Note**: this is *not* the most efficient implementation of partial
    gradient, however, the full gradient is computationally efficient.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of the circuit.
        x_vec: left-hand side vector, one of the sketched vectors.
        vh_y_vec: right-hand side vector, expects vh_y_vec = ``V.H @ y``,
                  **not** a ``y`` itself.
        workspace: array of temporary vectors of shape (>=3, x_vec.size).
        block_range: couple of indices [from, to) that defines a range of blocks
                     to compute gradients for; gradients of other blocks will be
                     set to zero; default value (None) implies full range.
        front_layer: flag enables gradient computation of the front layer
                     of 1-qubit gates.

    Returns:
        partial gradient of <V @ x, y> by theta parameters.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
    assert chk.complex_1d(x_vec)
    assert chk.complex_1d(vh_y_vec)
    assert chk.complex_2d(workspace, workspace.shape[0] >= 3)
    assert chk.no_overlap(x_vec, workspace)
    assert chk.no_overlap(vh_y_vec, workspace)
    assert chk.is_bool(front_layer)
    block_range = (0, circ.num_blocks) if block_range is None else block_range
    assert chk.is_tuple(block_range, len(block_range) == 2)
    assert 0 <= block_range[0] < block_range[1] <= circ.num_blocks

    b2b = bit2bit_transform  # shorthand alias
    n = circ.num_qubits
    blocks = circ.blocks
    trotterized = isinstance(circ, TrotterAnsatz)

    # Add a half-layer at the end of the 2nd order Trotter circuit.
    last_half_layer_num_blocks = int(0)
    if isinstance(circ, TrotterAnsatz) and circ.is_second_order:
        last_half_layer_num_blocks = circ.half_layer_num_blocks

    temp = workspace[0]  # temporary vector for intermediate results
    np.copyto(workspace[1], x_vec)
    np.copyto(workspace[2], vh_y_vec)
    w_vec = workspace[1]  # vector w, initially w = x
    z_vec = workspace[2]  # vector z, initially z = V.H @ y

    # Convenient split into parameter sub-sets for 1- and 2-qubit gates.
    grad = np.zeros(circ.num_thetas, dtype=np.cfloat)
    thetas1q, grad1q = circ.subset1q(thetas), circ.subset1q(grad)
    thetas2q, grad2q = circ.subset2q(thetas), circ.subset2q(grad)

    # Entangling gates are: CX, CZ or CP. Swappable gates are: Rx (X) or Rz (Z).
    cp_entangler = False
    if circ.entangler == "cp":
        cp_entangler = True
        entangling_gate_mul_vec = cp_mul_vec
        rs_mul_vec = rz_mul_vec
        dot_s = dot_z
    elif circ.entangler == "cz":
        entangling_gate_mul_vec = cz_mul_vec
        rs_mul_vec = rz_mul_vec
        dot_s = dot_z
    else:  # cx
        entangling_gate_mul_vec = cx_mul_vec
        rs_mul_vec = rx_mul_vec
        dot_s = dot_x

    # Computation for the front layer of 1-qubit gates.
    if front_layer:
        # Compute the derivatives.
        for q in range(n):
            pos = b2b(n, q)  # qubit position
            tht = thetas1q[q]

            rz_mul_vec(n, pos, tht[2], w_vec)  # w = Rz @ w
            rz_mul_vec(n, pos, tht[2], z_vec)  # z = Rz @ z
            grad1q[q, 2] = dot_z(n, pos, w_vec, z_vec, temp)  # 0.5j * <Z@w|z>

            ry_mul_vec(n, pos, tht[1], w_vec, temp)  # w = Ry @ w
            ry_mul_vec(n, pos, tht[1], z_vec, temp)  # z = Ry @ z
            grad1q[q, 1] = dot_y(n, pos, w_vec, z_vec, temp)  # 0.5j * <Y@w|z>

            rz_mul_vec(n, pos, tht[0], w_vec)  # w = Rz @ w
            rz_mul_vec(n, pos, tht[0], z_vec)  # z = Rz @ z
            grad1q[q, 0] = dot_z(n, pos, w_vec, z_vec, temp)  # 0.5j * <Z@w|z>
    else:
        # Skip computation of the derivatives.
        for q in range(n):
            pos = b2b(n, q)  # qubit position
            tht = thetas1q[q]

            rz_mul_vec(n, pos, tht[2], w_vec)  # w = Rz @ w
            rz_mul_vec(n, pos, tht[2], z_vec)  # z = Rz @ z

            ry_mul_vec(n, pos, tht[1], w_vec, temp)  # w = Ry @ w
            ry_mul_vec(n, pos, tht[1], z_vec, temp)  # z = Ry @ z

            rz_mul_vec(n, pos, tht[0], w_vec)  # w = Rz @ w
            rz_mul_vec(n, pos, tht[0], z_vec)  # z = Rz @ z

    # Compute derivatives of the 2-qubit blocks.
    #   Recall, "num_blocks" counts the number of blocks in full layers.
    # The latter does not include the blocks of the half-layer at the end,
    # in case of 2nd order Trotter circuit. Extra half-layer will be set
    # identical to the front one.
    for i in range(circ.num_blocks + last_half_layer_num_blocks):
        i_mod_nb = i % circ.num_blocks
        ctrl = b2b(n, blocks[0, i_mod_nb])  # control qubit
        targ = b2b(n, blocks[1, i_mod_nb])  # target qubit
        tht = thetas2q[i_mod_nb]

        if trotterized and i % 3 == 0:  # block triplet starts with Rz
            rz_mul_vec(n, ctrl, -np.pi / 2, w_vec)  # w = Rz @ w
            rz_mul_vec(n, ctrl, -np.pi / 2, z_vec)  # z = Rz @ z

        # Compute derivatives of the 2-qubit blocks inside the range. Note,
        # we write "grad2q[] += ..." because gradients of the front and trail
        # half-layers must be summed up in case of 2nd order Trotter.
        if block_range[0] <= i_mod_nb < block_range[1]:
            # Apply the entangling 2-qubit gate and compute its derivative.
            angle = float(tht[4]) if cp_entangler else 0.0
            entangling_gate_mul_vec(n, ctrl, targ, angle, z_vec, temp)  # z = E @ z
            if cp_entangler:  # CPhase has a parameter, so a bit more complicated
                derv_cphase_mul_vec(n, ctrl, targ, angle, w_vec, temp)
                grad2q[i_mod_nb, 4] += np.vdot(temp, z_vec).item()
            entangling_gate_mul_vec(n, ctrl, targ, angle, w_vec, temp)  # w = E @ w

            # Apply 1-qubit gates to control qubit.
            ry_mul_vec(n, ctrl, tht[0], w_vec, temp)  # w = Ry @ w
            ry_mul_vec(n, ctrl, tht[0], z_vec, temp)  # z = Ry @ z
            grad2q[i_mod_nb, 0] += dot_y(n, ctrl, w_vec, z_vec, temp)  # 0.5j * <Y@w|z>

            rz_mul_vec(n, ctrl, tht[1], w_vec)  # w = Rz @ w
            rz_mul_vec(n, ctrl, tht[1], z_vec)  # z = Rz @ z
            grad2q[i_mod_nb, 1] += dot_z(n, ctrl, w_vec, z_vec, temp)  # 0.5j * <Z@w|z>

            # Apply 1-qubit gates to target qubit. Mind the Swappable gate.
            ry_mul_vec(n, targ, tht[2], w_vec, temp)  # w = Ry @ w
            ry_mul_vec(n, targ, tht[2], z_vec, temp)  # z = Ry @ z
            grad2q[i_mod_nb, 2] += dot_y(n, targ, w_vec, z_vec, temp)  # 0.5j * <Y@w|z>

            rs_mul_vec(n, targ, tht[3], w_vec, temp)  # w = Rs @ w
            rs_mul_vec(n, targ, tht[3], z_vec, temp)  # z = Rs @ z
            grad2q[i_mod_nb, 3] += dot_s(n, targ, w_vec, z_vec, temp)  # 0.5j * <S@w|z>

        # Zero derivatives for the 2-qubit blocks outside the range:
        else:
            # Apply the entangling 2-qubit gate.
            angle = float(tht[4]) if cp_entangler else 0.0
            entangling_gate_mul_vec(n, ctrl, targ, angle, z_vec, temp)  # z = E @ z
            entangling_gate_mul_vec(n, ctrl, targ, angle, w_vec, temp)  # w = E @ w

            # Apply 1-qubit gates to control qubit.
            ry_mul_vec(n, ctrl, tht[0], w_vec, temp)  # w = Ry @ w
            ry_mul_vec(n, ctrl, tht[0], z_vec, temp)  # z = Ry @ z
            rz_mul_vec(n, ctrl, tht[1], w_vec)  # w = Rz @ w
            rz_mul_vec(n, ctrl, tht[1], z_vec)  # z = Rz @ z

            # Apply 1-qubit gates to target qubit. Mind the Swappable gates.
            ry_mul_vec(n, targ, tht[2], w_vec, temp)  # w = Ry @ w
            ry_mul_vec(n, targ, tht[2], z_vec, temp)  # z = Ry @ z
            rs_mul_vec(n, targ, tht[3], w_vec, temp)  # w = Rs @ w
            rs_mul_vec(n, targ, tht[3], z_vec, temp)  # z = Rs @ z

        if trotterized and i % 3 == 2:  # block triplet ends with Rz
            rz_mul_vec(n, targ, np.pi / 2, w_vec)  # w = Rz @ w
            rz_mul_vec(n, targ, np.pi / 2, z_vec)  # z = Rz @ z

    return grad
