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
Functions that convert parametric circuit into Qiskit quantum circuit
or Numpy matrix.

**Important**: in Numpy implementation we flipped bit ordering
to conform to Qiskit convention.
"""

from typing import Optional, Callable
import numpy as np
from qiskit import QuantumCircuit
import qiskit.quantum_info as qinfo
import aqc_research.checking as chk
from aqc_research.parametric_circuit import ParametricCircuit, TrotterAnsatz
from aqc_research.core_op_matrix import v_mul_mat
import aqc_research.core_operations as cop
from aqc_research.elementary_operations import (
    np_rx,
    np_ry,
    np_rz,
    np_phase,
    np_x,
    np_z,
    np_block_matrix,
)


def qcircuit_to_state(qc: QuantumCircuit) -> np.ndarray:
    """
    Returns the quantum state: ``|state> = circuit @ |0>``. **Note**,
    the routine becomes exponentially slow as the number of qubits grows.

    Args:
        qc: quantum circuit acting on the state ``|0>``.

    Returns:
        ``state vector == circuit @ |0>``.
    """
    assert isinstance(qc, QuantumCircuit) and qc.num_qubits >= 2
    return qinfo.Statevector(qc).data


def qcircuit_to_matrix(qc: QuantumCircuit) -> np.ndarray:
    """
    Converts quantum circuit to the corresponding matrix. **Note**,
    the routine becomes exponentially slow as the number of qubits grows.

    Args:
        qc: quantum circuit to build a corresponding matrix for.

    Returns:
        circuit matrix.
    """
    assert isinstance(qc, QuantumCircuit) and qc.num_qubits >= 2
    return qinfo.Operator(qc).data


def state_preparation_qcircuit(
    num_qubits: int,
    *,
    flip_bit: int = -1,
    state_prep_func: Optional[Callable[[int], QuantumCircuit]] = None,
) -> QuantumCircuit:
    """
    Returns quantum circuit that prepares either state ``S |0>`` or ``S X_i |0>``
    or just ``|0>``, where ``S`` is a low-entangling, state preparation circuit
    and ``X_i`` is an X gate applied to i-th qubit.

    Args:
        num_qubits: number of qubits.
        flip_bit: index of qubit where X gate will be applied prior to
                  derivative circuit; bit flip is needed for state preparation
                  with local Hilbert-Schmidt terms in objective function;
                  default behavior (flip_bit < 0) is doing nothing.
        state_prep_func: function for state preparation ``S``.

    Returns:
        quantum circuit that prepares a state from ``|0>``.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(flip_bit, flip_bit < num_qubits)
    assert state_prep_func is None or callable(state_prep_func)

    qc = QuantumCircuit(num_qubits)
    if flip_bit >= 0:
        qc.x(flip_bit)
    if callable(state_prep_func):
        qc = qc.compose(state_prep_func(num_qubits))
    return qc


def ansatz_to_qcircuit(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    qc: Optional[QuantumCircuit] = None,
    *,
    tol: Optional[float] = 0.0,
    clean_qc: Optional[bool] = False,
    visual: Optional[bool] = False,
) -> QuantumCircuit:
    """
    Constructs a Qiskit quantum circuit from the parametric ansatz and
    corresponding angular parameters. If a parameter value is less in absolute
    value than the specified tolerance then the corresponding rotation gate
    will be skipped in the output circuit.

    **Note**: the input quantum circuit will be modified in-place, if specified,
              however, we do not change its global phase (assuming global phase
              equals zero by default).

    **Note**: the input quantum circuit ``qc`` may already have some gates and
              non-zero global phase; one can set ``clean_qc=True`` the get
              rid of the old stuff, otherwise ansatz will be appended to ``qc``.

    **Note**: this function supports both Trotterized and generic ansatz layouts.


    Args:
        circ: parametric circuit (ansatz).
        thetas: angular parameters of circuit gates.
        qc: instance of QuantumCircuit that will be modified at the end.
        tol: tolerance level to identify nearly-zero angles; a gate will be
             discarded if having angular parameter close to zero.
        clean_qc: flag enables removal of all previously added gates from
                  quantum circuit and setting global phase to zero.
        visual: flag enables ``barrier`` separators for better visualization
                of Trotterized ansatz.

    Returns:
        modified instance of quantum circuit passed in ``qc'' argument.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
    assert qc is None or isinstance(qc, QuantumCircuit)
    assert chk.is_float(tol) and chk.is_bool(clean_qc) and chk.is_bool(visual)

    if circ.num_thetas != thetas.size:
        raise ValueError(
            f"wrong size of the vector of angular parameters, "
            f"expects: {circ.num_thetas}, got: {thetas.size}"
        )

    n = circ.num_qubits
    blocks = circ.blocks
    num_blocks = circ.num_blocks
    thetas1q = circ.subset1q(thetas)
    thetas2q = circ.subset2q(thetas)
    trotterized = isinstance(circ, TrotterAnsatz)
    visual = visual and trotterized

    # Add a half-layer at the end of the 2nd order Trotter circuit.
    last_half_layer_num_blocks = int(0)
    if isinstance(circ, TrotterAnsatz) and circ.is_second_order:
        last_half_layer_num_blocks = circ.half_layer_num_blocks

    if qc is None:
        qc = QuantumCircuit(n)
    elif clean_qc:  # remove all the gates in the qc
        qc.data.clear()
        qc.global_phase = float(0)

    # Entangling 2-qubit gates are: CX, CZ or CP. Note, the second 1-qubit
    # gate on the target qubit must be swappable with entangling gate.
    if circ.entangler == "cp":

        def entangler(_th: np.ndarray, _q1: int, _q2: int):
            qc.cp(_th[4], _q1, _q2)

        def swappable(_th: float, _q2: int):
            qc.rz(_th, _q2)

    elif circ.entangler == "cz":

        def entangler(_, _q1: int, _q2: int):
            qc.cz(_q1, _q2)

        def swappable(_th: float, _q2: int):
            qc.rz(_th, _q2)

    else:  # cx

        def entangler(_, _q1: int, _q2: int):
            qc.cx(_q1, _q2)

        def swappable(_th: float, _q2: int):
            qc.rx(_th, _q2)

    # In case circuit power > 1, we repeat circuit ``power`` times.
    for _ in range(circ.circuit_power):
        # Add initial rotation gates for each qubit (aka front gates).
        for q in range(n):
            t1q = thetas1q[q]
            if abs(t1q[2]) > tol:
                qc.rz(t1q[2], q)
            if abs(t1q[1]) > tol:
                qc.ry(t1q[1], q)
            if abs(t1q[0]) > tol:
                qc.rz(t1q[0], q)

        # Add 2-qubit unit blocks. Recall, "num_blocks" counts the number of
        # blocks in full layers. The latter does not include the blocks of the
        # half-layer at the end, in case of 2nd order Trotter circuit. Extra
        # half-layer will be set identical to the front one.
        for k in range(num_blocks + last_half_layer_num_blocks):
            if visual:
                _k2 = k % (3 * (n - 1))
                if _k2 == 0 or (k > 0 and _k2 == 3 * (n // 2)):
                    qc.barrier()

            k_mod_nb = k % num_blocks
            ctrl = int(blocks[0, k_mod_nb])  # control qubit
            targ = int(blocks[1, k_mod_nb])  # target qubit
            t2q = thetas2q[k_mod_nb]

            # In Trotterized ansatz a block triplet starts with Rz(-pi/2).
            if trotterized and k % 3 == 0:
                qc.rz(-np.pi / 2, ctrl)

            entangler(t2q, ctrl, targ)  # <-- CX, CZ or CP
            if abs(t2q[0]) > tol:
                qc.ry(t2q[0], ctrl)
            if abs(t2q[1]) > tol:
                qc.rz(t2q[1], ctrl)
            if abs(t2q[2]) > tol:
                qc.ry(t2q[2], targ)
            if abs(t2q[3]) > tol:
                swappable(t2q[3], targ)  # <-- Rx if CX, Rz if CP or CZ

            # In Trotterized ansatz a block triplet ends with Rz(pi/2).
            if trotterized and k % 3 == 2:
                qc.rz(np.pi / 2, targ)

    qc.name = circ.name
    return qc


def ansatz_to_numpy_by_qiskit(
    circ: ParametricCircuit, thetas: np.ndarray, tol: Optional[float] = 0.0
) -> np.ndarray:
    """
    Returns circuit matrix as Numpy array given angular parameters.
    Conversion is done by transformation into Qiskit circuit and then
    to Numpy matrix.

    **Note**: explicit construction of Numpy matrix can hit memory limit,
              mind the (exponential) problem size. This function is useful for
              debugging and testing, rather than for production.

    Args:
        circ: parametric circuit as defined in AQC framework.
        thetas: angular parameters of circuit gates.
        tol: tolerance level to identify nearly-zero angles.

    Returns:
        circuit matrix.
    """
    qc = QuantumCircuit(circ.num_qubits)
    return qcircuit_to_matrix(ansatz_to_qcircuit(circ, thetas, qc, tol=tol))


def ansatz_to_numpy_fast(circ: ParametricCircuit, thetas: np.ndarray) -> np.ndarray:
    """
    Returns circuit matrix as Numpy array given angular parameters.
    Similar to ``ansatz_to_numpy_by_qiskit()`` but does not use Qiskit at all.
    Instead, it builds circuit matrix directly with Numpy.

    **Note**, this function does *NOT* support the Trotterized ansatz of type
    ``TrotterAnsatz``. It was designed for generic ansatz with intention to
    accelerate the full AQC framework (based on matrix-matrix multiplications)
    rather than the state preparation/evolution (based on matrix-vector operations).
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas)
    mat = np.eye(circ.dimension, dtype=np.cfloat)
    return v_mul_mat(circ, thetas, mat, workspace=np.zeros_like(mat))


def ansatz_to_numpy_trotter(circ: ParametricCircuit, thetas: np.ndarray) -> np.ndarray:
    """
    Returns circuit matrix as Numpy array given angular parameters.
    Similar to ``ansatz_to_numpy_by_qiskit()`` but does not use Qiskit at all.
    Instead, it builds circuit matrix directly with Numpy.

    **Note**, this is not an efficient implementation. We keep it for
    illustration, completeness and as a reference for unit tests.

    **Note**, the suffix ``_trotter`` stands for the fact that this function
    *does* support the Trotterized ansatz of type ``TrotterAnsatz``.
    """
    assert isinstance(circ, ParametricCircuit)
    assert chk.float_1d(thetas)

    b2b = cop.bit2bit_transform  # shorthand alias
    qiskit_convention = bool(b2b(circ.num_qubits, 0) != 0)

    n = circ.num_qubits
    blocks = circ.blocks
    thetas1q = circ.subset1q(thetas)
    thetas2q = circ.subset2q(thetas)
    trotterized = isinstance(circ, TrotterAnsatz)

    # Add a half-layer at the end of the 2nd order Trotter circuit.
    last_half_layer_num_blocks = int(0)
    if isinstance(circ, TrotterAnsatz) and circ.is_second_order:
        last_half_layer_num_blocks = circ.half_layer_num_blocks

    # Entangling 2-qubit gates are: CX, CZ or CP. Note, the second 1-qubit
    # gate on the target qubit must be swappable with entangling gate.
    x_gate, z_gate = np_x(), np_z()
    if circ.entangler == "cp":

        def ctrl_gate(_th: np.ndarray) -> np.ndarray:
            return np_phase(_th[4])

        def swappable(_th: float):
            return np_rz(_th)

    elif circ.entangler == "cz":

        def ctrl_gate(_) -> np.ndarray:
            return z_gate

        def swappable(_th: float):
            return np_rz(_th)

    else:

        def ctrl_gate(_) -> np.ndarray:
            return x_gate

        def swappable(_th: float):
            return np_rx(_th)

    # Add 2-qubit unit blocks. Recall, "num_blocks" counts the number of
    # blocks in full layers. The latter does not include the blocks of the
    # half-layer at the end, in case of 2nd order Trotter circuit. Extra
    # half-layer will be set identical to the front one.
    v_mat = np.eye(circ.dimension)
    for k in range(circ.num_blocks + last_half_layer_num_blocks):
        k_mod_nb = k % circ.num_blocks
        ctrl = b2b(n, blocks[0, k_mod_nb])  # control qubit
        targ = b2b(n, blocks[1, k_mod_nb])  # target qubit
        t2q = thetas2q[k_mod_nb]

        b_mat = np_block_matrix(
            n,
            ctrl,  # control qubit
            targ,  # target qubit
            np_rz(t2q[1]) @ np_ry(t2q[0]),
            swappable(t2q[3]) @ np_ry(t2q[2]),
            ctrl_gate(t2q),
        )

        # In Trotterized ansatz a block triplet starts and ends with Rz.
        if trotterized:
            if k % 3 == 0:
                eye1, eye2 = np.eye(2**ctrl), np.eye(2 ** (n - 1 - ctrl))
                b_mat = b_mat @ np.kron(np.kron(eye1, np_rz(-np.pi / 2)), eye2)
            if k % 3 == 2:
                eye1, eye2 = np.eye(2**targ), np.eye(2 ** (n - 1 - targ))
                b_mat = np.kron(np.kron(eye1, np_rz(np.pi / 2)), eye2) @ b_mat

        v_mat = b_mat @ v_mat

    # Add the front rotation gates ZYZ. Mind the order of matrix multiplication
    # which depends on bit ordering.
    v1_mat = 1
    if qiskit_convention:
        for k in range(n):
            t1q = thetas1q[k]
            v1_mat = np.kron(np_rz(t1q[0]) @ np_ry(t1q[1]) @ np_rz(t1q[2]), v1_mat)
    else:
        for k in range(n):
            t1q = thetas1q[k]
            v1_mat = np.kron(v1_mat, np_rz(t1q[0]) @ np_ry(t1q[1]) @ np_rz(t1q[2]))

    v_mat = v_mat @ v1_mat
    return v_mat
