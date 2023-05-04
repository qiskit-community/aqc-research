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
Utilities for computation of objective function and its gradient given the
objective in the form of dot product. State vectors are assumed in MPS format.
"""
# Method had been moved in AirSimulator in the latest Qiskit. FIXME
# Instance of 'QuantumCircuit' has no 'set_matrix_product_state' memberPylint(E1101:no-member)

import copy
from typing import Optional, Tuple, Callable
from functools import partial
import numpy as np
from qiskit import QuantumCircuit
from aqc_research.parametric_circuit import ParametricCircuit, TrotterAnsatz
import aqc_research.checking as chk
from aqc_research.mps_operations import (
    QiskitMPS,
    mps_from_circuit,
    mps_dot,
    no_truncation_threshold,
    check_mps,
)


# -----------------------------------------------------------------------------
# Fast MPS gradient implementation.
# -----------------------------------------------------------------------------


def fast_dot_gradient(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    lvec: QiskitMPS,
    vh_phi: QiskitMPS,
    *,
    trunc_thr: Optional[float] = no_truncation_threshold(),
    block_range: Optional[Tuple[int, int]] = None,
    front_layer: Optional[bool] = True,
) -> np.ndarray:
    """
    Computes fast gradient of the dot product ``<lvec|V.H|phi>`` using MPS, where
    ``V`` is a **generic** parametric circuit (ansatz), ``|phi>`` is a target
    state (in MPS format), and ``|lvec>`` is a low-entangled left-hand side vector,
    typically ``|0>`` (also in MPS format).

    Gradients are computed for the front layer of 1-qubit gates, if enabled,
    and for 2-qubit unit blocks in the range ``[block_range[0], block_range[1])``
    (mind semi-open interval). For the blocks outside the range this function
    will return zero gradients. Default values in the last two arguments imply
    computation of the full gradient.

    **Note**: It is assumed that initially (!) ``vh_phi = V.H @ |phi>``,
    which is a byproduct of computation of the objective function prior
    to calling this one. We do so because Qiskit optimizers separates
    computation of objective function and its gradient. By reusing
    ``V.H @ |phi>`` obtained after objective function, we save a fair amount
    of computations.

    **Note**: this is *not* the most efficient implementation of partial
    gradient, however, the full gradient is computationally efficient.

    Args:
        circ: instance of parametric circuit.
        thetas: angular parameters of the circuit.
        lvec: left-hand side vector in MPS format.
        vh_phi: right-hand side vector in MPS format, expects
                vh_phi = ``V.H @ |phi>``, **not** a ``|phi>`` itself.
        trunc_thr: truncation threshold in MPS representation.
        block_range: couple of indices [from, to) that defines a range of blocks
                     to compute gradients for; gradients of other blocks will be
                     set to zero; default value (None) implies full range.
        front_layer: flag enables gradient computation of the front layer
                     of 1-qubit gates.

    Returns:
        vector of complex gradients.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas)
    assert isinstance(lvec, tuple) and isinstance(vh_phi, tuple)
    assert chk.is_float(trunc_thr, trunc_thr >= 0)
    assert chk.is_bool(front_layer)
    block_range = (0, circ.num_blocks) if block_range is None else block_range
    assert chk.is_tuple(block_range, len(block_range) == 2)
    assert 0 <= block_range[0] < block_range[1] <= circ.num_blocks

    n = circ.num_qubits
    blocks = circ.blocks
    w_vec = copy.deepcopy(lvec)  # deep copy!
    z_vec = copy.deepcopy(vh_phi)
    trotterized = isinstance(circ, TrotterAnsatz)

    # Add a half-layer at the end of the 2nd order Trotter circuit.
    last_half_layer_num_blocks = int(0)
    if isinstance(circ, TrotterAnsatz) and circ.is_second_order:
        last_half_layer_num_blocks = circ.half_layer_num_blocks

    # Convenient split into parameter sub-sets for 1- and 2-qubit gates.
    grad = np.zeros(circ.num_thetas, dtype=np.cfloat)
    thetas1q, grad1q = circ.subset1q(thetas), circ.subset1q(grad)
    thetas2q, grad2q = circ.subset2q(thetas), circ.subset2q(grad)

    # Entangling gates are: CX, CZ or CP. Swappable gates are: Rx (X) or Rz (Z).
    cp_entangler = False
    if circ.entangler == "cp":
        cp_entangler = True
        entangler_mul_mps = partial(cp_mul_mps, trunc_thr=trunc_thr)
        rs_mul_mps = rz_mul_mps
        dot_s = dot_z
    elif circ.entangler == "cz":
        entangler_mul_mps = partial(cz_mul_mps, trunc_thr=trunc_thr)
        rs_mul_mps = rz_mul_mps
        dot_s = dot_z
    else:  # cx
        entangler_mul_mps = partial(cx_mul_mps, trunc_thr=trunc_thr)
        rs_mul_mps = rx_mul_mps
        dot_s = dot_x

    # Computation for the front layer of 1-qubit gates.
    if front_layer:
        # Compute the derivatives.
        for q in range(n):
            tht = thetas1q[q]

            w_vec = rz_mul_mps(tht[2], q, w_vec)  # w = Rz @ w
            z_vec = rz_mul_mps(tht[2], q, z_vec)  # z = Rz @ z
            grad1q[q, 2] = dot_z(q, w_vec, z_vec)  # 0.5j * <Z@w|z>

            w_vec = ry_mul_mps(tht[1], q, w_vec)  # w = Ry @ w
            z_vec = ry_mul_mps(tht[1], q, z_vec)  # z = Ry @ z
            grad1q[q, 1] = dot_y(q, w_vec, z_vec)  # 0.5j * <Y@w|z>

            w_vec = rz_mul_mps(tht[0], q, w_vec)  # w = Rz @ w
            z_vec = rz_mul_mps(tht[0], q, z_vec)  # z = Rz @ z
            grad1q[q, 0] = dot_z(q, w_vec, z_vec)  # 0.5j * <Z@w|z>
    else:
        # Skip computation of the derivatives.
        for q in range(n):
            tht = thetas1q[q]

            w_vec = rz_mul_mps(tht[2], q, w_vec)  # w = Rz @ w
            z_vec = rz_mul_mps(tht[2], q, z_vec)  # z = Rz @ z

            w_vec = ry_mul_mps(tht[1], q, w_vec)  # w = Ry @ w
            z_vec = ry_mul_mps(tht[1], q, z_vec)  # z = Ry @ z

            w_vec = rz_mul_mps(tht[0], q, w_vec)  # w = Rz @ w
            z_vec = rz_mul_mps(tht[0], q, z_vec)  # z = Rz @ z

    # Compute derivatives of the 2-qubit blocks.
    #   Recall, "num_blocks" counts the number of blocks in full layers.
    # The latter does not include the blocks of the half-layer at the end,
    # in case of 2nd order Trotter circuit. Extra half-layer will be set
    # identical to the front one.
    for i in range(circ.num_blocks + last_half_layer_num_blocks):
        i_mod_nb = i % circ.num_blocks
        ctrl = blocks[0, i_mod_nb]  # control qubit
        targ = blocks[1, i_mod_nb]  # target qubit
        tht = thetas2q[i_mod_nb]

        if trotterized and i % 3 == 0:  # block triplet starts with Rz
            w_vec = rz_mul_mps(-np.pi / 2, ctrl, w_vec)  # w = Rz @ w
            z_vec = rz_mul_mps(-np.pi / 2, ctrl, z_vec)  # z = Rz @ z

        # Compute derivatives of the 2-qubit blocks inside the range. Note,
        # we write "grad2q[] += ..." because gradients of the front and rear
        # half-layers must be summed up in case of 2nd order Trotter.
        if block_range[0] <= i_mod_nb < block_range[1]:
            # Apply the entangling 2-qubit gate and compute its derivative.
            # CPhase has a parameter, so a bit more complicated. Namely,
            # derivative of CPhase gate in not proportional to a unitary
            # (in contrast to Rx, Ry, Rx). Instead, we take the derivative as
            # a difference of two CPhase gates with shifted parameter and
            # compute the dot products separately for each one of them.
            if cp_entangler:
                angle = float(tht[4])
                angle2 = angle + np.pi

                z_vec = entangler_mul_mps(angle, ctrl, targ, z_vec)  # z = E @ z
                w_vec2 = entangler_mul_mps(angle2, ctrl, targ, w_vec)  # w2 = E2 @ w
                w_vec = entangler_mul_mps(angle, ctrl, targ, w_vec)  # w = E @ w

                cp_w_z = mps_dot(w_vec, z_vec)  # <(cp(a) @ w)|z>
                cp_w_z2 = mps_dot(w_vec2, z_vec)  # <(cp(a+pi) @ w)|z>
                grad2q[i_mod_nb, 4] += np.cfloat(-0.5j * (cp_w_z - cp_w_z2))
            else:
                z_vec = entangler_mul_mps(0.0, ctrl, targ, z_vec)  # z = E @ z
                w_vec = entangler_mul_mps(0.0, ctrl, targ, w_vec)  # w = E @ w

            # Apply 1-qubit gates to control qubit.
            w_vec = ry_mul_mps(tht[0], ctrl, w_vec)  # w = Ry @ w
            z_vec = ry_mul_mps(tht[0], ctrl, z_vec)  # z = Ry @ z
            grad2q[i_mod_nb, 0] += dot_y(ctrl, w_vec, z_vec)  # 0.5j * <Y@w|z>

            w_vec = rz_mul_mps(tht[1], ctrl, w_vec)  # w = Rz @ w
            z_vec = rz_mul_mps(tht[1], ctrl, z_vec)  # z = Rz @ z
            grad2q[i_mod_nb, 1] += dot_z(ctrl, w_vec, z_vec)  # 0.5j * <Z@w|z>

            # Apply 1-qubit gates to target qubit. Mind the Swappable gate.
            w_vec = ry_mul_mps(tht[2], targ, w_vec)  # w = Ry @ w
            z_vec = ry_mul_mps(tht[2], targ, z_vec)  # z = Ry @ z
            grad2q[i_mod_nb, 2] += dot_y(targ, w_vec, z_vec)  # 0.5j * <Y@w|z>

            w_vec = rs_mul_mps(tht[3], targ, w_vec)  # w = Rs @ w
            z_vec = rs_mul_mps(tht[3], targ, z_vec)  # z = Rs @ z
            grad2q[i_mod_nb, 3] += dot_s(targ, w_vec, z_vec)  # 0.5j * <S@w|z>

        # Zero derivatives for the 2-qubit blocks outside the range:
        else:
            # Apply the entangling 2-qubit gate.
            angle = float(tht[4]) if cp_entangler else 0.0
            z_vec = entangler_mul_mps(angle, ctrl, targ, z_vec)  # z = E @ z
            w_vec = entangler_mul_mps(angle, ctrl, targ, w_vec)  # w = E @ w

            # Apply 1-qubit gates to control qubit.
            w_vec = ry_mul_mps(tht[0], ctrl, w_vec)  # w = Ry @ w
            z_vec = ry_mul_mps(tht[0], ctrl, z_vec)  # z = Ry @ z
            w_vec = rz_mul_mps(tht[1], ctrl, w_vec)  # w = Rz @ w
            z_vec = rz_mul_mps(tht[1], ctrl, z_vec)  # z = Rz @ z

            # Apply 1-qubit gates to target qubit. Mind the Swappable gate.
            w_vec = ry_mul_mps(tht[2], targ, w_vec)  # w = Ry @ w
            z_vec = ry_mul_mps(tht[2], targ, z_vec)  # z = Ry @ z
            w_vec = rs_mul_mps(tht[3], targ, w_vec)  # w = Rs @ w
            z_vec = rs_mul_mps(tht[3], targ, z_vec)  # z = Rs @ z

        if trotterized and i % 3 == 2:  # block triplet ends with Rz
            w_vec = rz_mul_mps(np.pi / 2, targ, w_vec)  # w = Rz @ w
            z_vec = rz_mul_mps(np.pi / 2, targ, z_vec)  # z = Rz @ z

    return grad


def x_mul_mps(qubit: int, mps_vec: QiskitMPS) -> QiskitMPS:
    """
    Applies X-gate to right-hand side state vector in Qiskit MPS format
    and returns augmented MPS (of gate times vector product). Note, there is
    no truncation threshold among function arguments because multiplication
    by 1-qubit gate does not change bond dimension.

    Args:
        qubit: index of qubit where gate is applied.
        mps_vec: right-hand side vector to be multiplied by the circuit.

    Returns:
        Qiskit MPS data.
    """
    num_qubits = len(mps_vec[0])
    assert chk.is_int(qubit, 0 <= qubit < num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(mps_vec)
    qc.x(qubit)
    return mps_from_circuit(qc)


def y_mul_mps(qubit: int, mps_vec: QiskitMPS) -> QiskitMPS:
    """
    Applies Y-gate to right-hand side state vector in Qiskit MPS format
    and returns augmented MPS (of gate times vector product). Note, there is
    no truncation threshold among function arguments because multiplication
    by 1-qubit gate does not change bond dimension.

    Args:
        qubit: index of qubit where gate is applied.
        mps_vec: right-hand side vector to be multiplied by the circuit.

    Returns:
        Qiskit MPS data.
    """
    num_qubits = len(mps_vec[0])
    assert chk.is_int(qubit, 0 <= qubit < num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(mps_vec)
    qc.y(qubit)
    return mps_from_circuit(qc)


def z_mul_mps(qubit: int, mps_vec: QiskitMPS) -> QiskitMPS:
    """
    Applies Z gate to right-hand side state vector in Qiskit MPS format
    and returns augmented MPS (of gate times vector product). Note, there is
    no truncation threshold among function arguments because multiplication
    by 1-qubit gate does not change bond dimension.

    Args:
        qubit: index of qubit where gate is applied.
        mps_vec: right-hand side vector to be multiplied by the circuit.

    Returns:
        Qiskit MPS data.
    """
    num_qubits = len(mps_vec[0])
    assert chk.is_int(qubit, 0 <= qubit < num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(mps_vec)
    qc.z(qubit)
    return mps_from_circuit(qc)


def rx_mul_mps(angle: float, qubit: int, mps_vec: QiskitMPS) -> QiskitMPS:
    """
    Applies Rx gate to right-hand side state vector in Qiskit MPS format
    and returns augmented MPS (of gate times vector product). Note, there is
    no truncation threshold among function arguments because multiplication
    by 1-qubit gate does not change bond dimension.

    Args:
        angle: rotation angle.
        qubit: index of qubit where gate is applied.
        mps_vec: right-hand side vector to be multiplied by the circuit.

    Returns:
        Qiskit MPS data.
    """
    num_qubits = len(mps_vec[0])
    assert chk.is_int(qubit, 0 <= qubit < num_qubits) and chk.is_float(angle)
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(mps_vec)
    qc.rx(angle, qubit)
    return mps_from_circuit(qc)


def ry_mul_mps(angle: float, qubit: int, mps_vec: QiskitMPS) -> QiskitMPS:
    """
    Applies Ry gate to right-hand side state vector in Qiskit MPS format
    and returns augmented MPS (of gate times vector product). Note, there is
    no truncation threshold among function arguments because multiplication
    by 1-qubit gate does not change bond dimension.

    Args:
        angle: rotation angle.
        qubit: index of qubit where gate is applied.
        mps_vec: right-hand side vector to be multiplied by the circuit.

    Returns:
        Qiskit MPS data.
    """
    num_qubits = len(mps_vec[0])
    assert chk.is_int(qubit, 0 <= qubit < num_qubits) and chk.is_float(angle)
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(mps_vec)
    qc.ry(angle, qubit)
    return mps_from_circuit(qc)


def rz_mul_mps(angle: float, qubit: int, mps_vec: QiskitMPS) -> QiskitMPS:
    """
    Applies Rz gate to right-hand side state vector in Qiskit MPS format
    and returns augmented MPS (of gate times vector product). Note, there is
    no truncation threshold among function arguments because multiplication
    by 1-qubit gate does not change bond dimension.

    Args:
        angle: rotation angle.
        qubit: index of qubit where gate is applied.
        mps_vec: right-hand side vector to be multiplied by the circuit.

    Returns:
        Qiskit MPS data.
    """
    num_qubits = len(mps_vec[0])
    assert chk.is_int(qubit, 0 <= qubit < num_qubits) and chk.is_float(angle)
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(mps_vec)
    qc.rz(angle, qubit)
    return mps_from_circuit(qc)


def cx_mul_mps(
    _: float,
    ctrl: int,
    targ: int,
    mps_vec: QiskitMPS,
    *,
    trunc_thr: float = no_truncation_threshold(),
) -> QiskitMPS:
    """
    Applies CX gate to right-hand side state vector in Qiskit MPS format
    and returns augmented MPS (of gate times vector product).

    Args:
        _: unused angular parameter; presents here for interface compatibility.
        ctrl: index of control qubit.
        targ: index of target qubit.
        mps_vec: right-hand side vector to be multiplied by the circuit.
        trunc_thr: truncation threshold in MPS representation.

    Returns:
        Qiskit MPS data.
    """
    num_qubits = len(mps_vec[0])
    assert 0 <= ctrl < num_qubits and 0 <= targ < num_qubits and ctrl != targ
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(mps_vec)
    qc.cx(ctrl, targ)
    return mps_from_circuit(qc, trunc_thr=trunc_thr)


def cp_mul_mps(
    angle: float,
    ctrl: int,
    targ: int,
    mps_vec: QiskitMPS,
    *,
    trunc_thr: float = no_truncation_threshold(),
) -> QiskitMPS:
    """
    Applies CPhase gate to right-hand side state vector in Qiskit MPS format
    and returns augmented MPS (of gate times vector product).

    Args:
        angle: rotation angle.
        ctrl: index of control qubit.
        targ: index of target qubit.
        mps_vec: right-hand side vector to be multiplied by the circuit.
        trunc_thr: truncation threshold in MPS representation.

    Returns:
        Qiskit MPS data.
    """
    num_qubits = len(mps_vec[0])
    assert 0 <= ctrl < num_qubits and 0 <= targ < num_qubits and ctrl != targ
    assert chk.is_float(angle)
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(mps_vec)
    qc.cp(angle, ctrl, targ)
    return mps_from_circuit(qc, trunc_thr=trunc_thr)


def cz_mul_mps(
    _: float,
    ctrl: int,
    targ: int,
    mps_vec: QiskitMPS,
    *,
    trunc_thr: float = no_truncation_threshold(),
) -> QiskitMPS:
    """
    Applies CZ gate to right-hand side state vector in Qiskit MPS format
    and returns augmented MPS (of gate times vector product).

    Args:
        _: unused angular parameter; presents here for interface compatibility.
        ctrl: index of control qubit.
        targ: index of target qubit.
        mps_vec: right-hand side vector to be multiplied by the circuit.
        trunc_thr: truncation threshold in MPS representation.

    Returns:
        Qiskit MPS data.
    """
    num_qubits = len(mps_vec[0])
    assert 0 <= ctrl < num_qubits and 0 <= targ < num_qubits and ctrl != targ
    qc = QuantumCircuit(num_qubits)
    qc.set_matrix_product_state(mps_vec)
    qc.cz(ctrl, targ)
    return mps_from_circuit(qc, trunc_thr=trunc_thr)


def dot_x(qubit: int, w_vec: QiskitMPS, z_vec: QiskitMPS) -> np.cfloat:
    """
    Computes 0.5j * <X@w|z>. Note, there is no truncation threshold among function
    arguments because multiplication by 1-qubit gate does not change bond dimension.

    Args:
        qubit: position of a qubit where 1-qubit X-gate is applied.
        w_vec: w vector.
        z_vec: y vector.

    Returns:
        0.5j * <X@w|z>.
    """
    return np.cfloat(0.5j * mps_dot(x_mul_mps(qubit, w_vec), z_vec))


def dot_y(qubit: int, w_vec: QiskitMPS, z_vec: QiskitMPS) -> np.cfloat:
    """
    Computes 0.5j * <Y@w|z>. Note, there is no truncation threshold among function
    arguments because multiplication by 1-qubit gate does not change bond dimension.

    Args:
        qubit: position of a qubit where 1-qubit Y-gate is applied.
        w_vec: w vector.
        z_vec: y vector.

    Returns:
        0.5j * <Y@w|z>.
    """
    return np.cfloat(0.5j * mps_dot(y_mul_mps(qubit, w_vec), z_vec))


def dot_z(qubit: int, w_vec: QiskitMPS, z_vec: QiskitMPS) -> np.cfloat:
    """
    Computes 0.5j * <Z@w|z>. Note, there is no truncation threshold among function
    arguments because multiplication by 1-qubit gate does not change bond dimension.

    Args:
        qubit: position of a qubit where 1-qubit Z-gate is applied.
        w_vec: w vector.
        z_vec: y vector.

    Returns:
        0.5j * <Z@w|z>.
    """
    return np.cfloat(0.5j * mps_dot(z_mul_mps(qubit, w_vec), z_vec))
