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
Utilities for manipulating the state vectors in MPS format.
"""
# Method had been moved in AirSimulator in the latest Qiskit. FIXME
# Instance of 'QuantumCircuit' has no 'set_matrix_product_state' memberPylint(E1101:no-member)

from typing import List, Optional, Tuple
from random import choice
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from aqc_research.parametric_circuit import ParametricCircuit
from aqc_research.circuit_transform import ansatz_to_qcircuit
from aqc_research.circuit_structures import create_ansatz_structure
import aqc_research.utils as helper
import aqc_research.checking as chk

_NO_TRUNCATION_THR = 1e-16

# Type of MPS data as it outputted by Qiskit.
QiskitMPS = Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]


# class MPSdata:
#     """Class keep *preprocessed* MPS data, see _preprocess_mps() for details."""
#
#     def __init__(
#         self,
#         mps_matrices: List[np.ndarray],
#         mps_phase: Optional[float] = 0.0,
#         state: Optional[np.ndarray] = None,
#     ):
#         assert chk.is_list(mps_matrices)
#         assert len(mps_matrices) == 0 or chk.complex_3d(mps_matrices[0])
#         assert chk.is_float(mps_phase)
#         assert state is None or chk.complex_1d(state)
#
#         self._matrices = mps_matrices
#         self._phase = mps_phase
#         self._state = state
#         assert self._is_consistent()
#
#     def _is_consistent(self) -> bool:
#         if self._state is None or len(self._matrices) == 0:
#             return True
#         return self._state.size == 2 ** len(self._matrices)
#
#     @property
#     def num_qubits(self) -> int:
#         if len(self.matrices) > 0:
#             return len(self._matrices)
#         elif self._state is not None and self._state.size > 0:
#             return int(round(np.log2(float(self._state.size))))
#         else:
#             return int(0)
#
#     @property
#     def matrices(self) -> List[np.ndarray]:
#         return self._matrices
#
#     @property
#     def phase(self) -> float:
#         return self._phase
#
#     @property
#     def state(self) -> Union[np.ndarray, None]:
#         return self._state


def no_truncation_threshold() -> float:
    """Returns "no truncation" threshold value for MPS computation."""
    return _NO_TRUNCATION_THR


def check_mps(qiskit_mps: QiskitMPS) -> bool:
    """
    Checks that the input argument is really an MPS decomposition produced
    by Qiskit framework.

    Args:
        qiskit_mps: state in MPS representation.

    Returns:
        True in case of valid input structure.
    """
    if not (isinstance(qiskit_mps, tuple) and len(qiskit_mps) == 2):
        return False  # expects MPS as a tuple of size 2

    num_qubits = len(qiskit_mps[0])
    gam, lam = qiskit_mps[0], qiskit_mps[1]  # Gamma and diagonal Lambda matrices
    if len(gam) != num_qubits:
        return False  # wrong number of Gamma matrices
    if len(lam) != num_qubits - 1:
        return False  # wrong number of Lambda matrices

    for n in range(num_qubits):
        g_n = gam[n]
        if g_n[0].ndim != 2:
            return False  # Gamma is not a matrix
        if len(g_n) != 2:
            return False  # expects 2 Gammas per qubit
        if g_n[0].shape != g_n[1].shape:
            return False  # unequal shapes of Gammas
        if n < num_qubits - 1:
            l_n = lam[n]
            if not (l_n.ndim == 1 or (l_n.ndim == 2 and min(l_n.shape) == 1)):
                return False  # expects vector input
            l_n = l_n.ravel()
            if not np.all(l_n[:-1] >= l_n[1:]):
                return False  # expects vector sorted in descending order
    return True


def _preprocess_mps(qiskit_mps: QiskitMPS, conjugate: bool = False) -> List[np.ndarray]:
    """
    Converts Qiskit MPS representation is more compact, custom one, which is
    better suitable for further computation. Namely, we combine Gamma matrices
    for both bit states 0/1 into a single 3D tensor and multiply it by the
    subsequent diagonal Lambda matrix (except the very last Gamma).

    Args:
        qiskit_mps: MPS decomposition of a quantum state as it comes out from
                    Qiskit framework.
        conjugate: optional conjugation of the output MPS tensors.

    Returns:
        list of size ``num_qubits`` of MPS tensors.
    """
    assert check_mps(qiskit_mps) and isinstance(conjugate, bool)
    num_qubits = len(qiskit_mps[0])
    gam, lam = qiskit_mps[0], qiskit_mps[1]  # Gamma and diagonal Lambda matrices
    my_mps = list([])
    for n in range(num_qubits):
        g_n = gam[n]
        g_n = np.stack((g_n[0], g_n[1]))  # combine into a single tensor
        if n < num_qubits - 1:
            g_n *= np.expand_dims(lam[n], (0, 1))  # gamma <-- gamma * lambda

        if conjugate:
            my_mps.append(np.conjugate(g_n, out=g_n))
        else:
            my_mps.append(g_n)

    return my_mps


def mps_to_vector(qiskit_mps: QiskitMPS) -> np.ndarray:
    """
    Computes coefficients ``coef`` of individual quantum basis states:
    ``coef_{i1,i2,...,in} * |i1> kron |i2> kron ... kron |in>``, and
    builds a state vector from MPS representation.

    **Note**, the operation is slow and time-consuming for a large number
    of qubits. It was designed primarily for testing.

    Args:
        qiskit_mps: MPS decomposition of the state produced by Qiskit framework.

    Returns:
        quantum state as a vector of size ``2^n``.
    """
    mps = _preprocess_mps(qiskit_mps)
    num_qubits = len(mps)
    state = np.zeros(2**num_qubits, dtype=np.cfloat)
    for k in range(state.size):  # for all combinations of individual bits ...
        coef = None
        for n in range(num_qubits):
            b = (k >> n) & 1  # k-th bit state (0/1) at qubit 'n'
            g_n = mps[n][b, ...]  # Gamma at qubit 'n'
            if coef is None:
                coef = g_n.copy()
            else:
                coef = np.tensordot(coef, g_n, axes=([1], [0]))

        state[k] = coef.item()  # must be a scalar at the end

    return state


def mps_dot(qiskit_mps1: QiskitMPS, qiskit_mps2: QiskitMPS) -> np.cfloat:
    """
    Computes dot product between MPS decompositions of two quantum states:
    ``< mps1 | mps2 >``.

    Args:
        qiskit_mps1: MPS decomposition of the left state.
        qiskit_mps2: MPS decomposition of the right state.

    Returns:
        complex dot product value.
    """
    mat1, mat2 = _preprocess_mps(qiskit_mps1), _preprocess_mps(qiskit_mps2)
    a, b = np.squeeze(mat1[0], axis=1), np.squeeze(mat2[0], axis=1)
    a_b = np.tensordot(np.conj(a), b, axes=([0], [0]))

    assert len(mat1) == len(mat2)  # same number of qubits
    for n in range(1, len(mat1)):
        a_b = np.tensordot(a_b, np.conj(mat1[n]), axes=([0], [1]))
        a_b = np.tensordot(a_b, mat2[n], axes=([0, 1], [1, 0]))

    return np.cfloat(a_b.item())


def mps_from_circuit(
    qc: QuantumCircuit,
    *,
    trunc_thr: Optional[float] = _NO_TRUNCATION_THR,
    out_state: Optional[np.ndarray] = None,
    print_log_data: Optional[bool] = False,
) -> QiskitMPS:
    """
    Computes MPS representation of output state (in Qiskit format) after quantum
    circuit acting on zero state: ``output = circuit @ |0>``.

    **Note**, this function modifies the input circuit in a way that one cannot
    apply it twice. Qiskit limitation: ``save_statevector()`` is applicable
    only once.

    Args:
        qc: quantum circuit that acts on state ``|0>``.
        trunc_thr: truncation threshold in MPS representation.
        out_state: output array for storing state as a normal vector; *note*,
                   state generation can be a slow and even intractable operation
                   for the large number of qubits; useful for testing only.
        print_log_data: flag enables printing of MPS internal information;
                        useful for debugging and testing.

    Returns:
        MPS state representation as outputted by Qiskit framework.
    """
    assert isinstance(qc, QuantumCircuit)
    assert chk.is_float(trunc_thr, 0 <= trunc_thr <= 0.1)
    assert isinstance(print_log_data, bool)

    if isinstance(out_state, np.ndarray):
        qc.save_statevector(label="my_sv")
        assert chk.complex_1d(out_state, out_state.size == 2**qc.num_qubits)

    qc.save_matrix_product_state(label="my_mps")
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_truncation_threshold=trunc_thr,
        mps_log_data=print_log_data,
    )
    result = sim.run(qc, shots=1).result()
    data = result.data(0)

    if print_log_data:
        print(result.results[0].metadata["MPS_log_data"])
    if isinstance(out_state, np.ndarray):
        np.copyto(out_state, np.asarray(data["my_sv"]))

    return data["my_mps"]


def qcircuit_mul_mps(
    qc: QuantumCircuit,
    mps_vec: QiskitMPS,
    *,
    trunc_thr: Optional[float] = _NO_TRUNCATION_THR,
    out_state: Optional[np.ndarray] = None,
) -> QiskitMPS:
    """
    Applies quantum circuit to right-hand side state vector given in MPS
    format and returns augmented MPS, i.e. circuit times vector product.

    Args:
        qc: quantum circuit to be applied to a vector in MPS format.
        mps_vec: right-hand side vector to be multiplied by the circuit.
        trunc_thr: truncation threshold in MPS representation.
        out_state: output array for storing state as a normal vector;
                   *note*, this can be very slow and even intractable
                   for a large number of qubits; useful for testing only.

    Returns:
        product of circuit and right-hand vector in MPS format.
    """
    assert isinstance(qc, QuantumCircuit)
    assert chk.is_tuple(mps_vec) and len(mps_vec[0]) == qc.num_qubits

    # Note, the order of operations is crucial. We fill in the circuit after (!)
    # invocation of set_matrix_product_state().
    qc2 = QuantumCircuit(qc.num_qubits)
    qc2.set_matrix_product_state(mps_vec)
    qc2.compose(qc, inplace=True)
    return mps_from_circuit(qc2, trunc_thr=trunc_thr, out_state=out_state)


def rand_mps_vec(
    num_qubits: int,
    out_state: Optional[np.ndarray] = None,
    num_layers: int = 3,
) -> QiskitMPS:
    """
    Generates a random vector in MPS format.

    Args:
        num_qubits: number of qubits.
        out_state: output array for storing state as a normal vector (slow!).
        num_layers: number of layers with (n-1) blocks each.

    Returns:
        (1) Qiskit MPS vector, (2) global phase, (3) state vector or None.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(num_layers, num_layers > 0)
    blocks = create_ansatz_structure(num_qubits, "spin", "full", num_layers * (num_qubits - 1))
    circ = ParametricCircuit(num_qubits, choice(["cx", "cz", "cp"]), blocks)
    thetas = helper.rand_thetas(circ.num_thetas)
    qc = ansatz_to_qcircuit(circ, thetas)
    return mps_from_circuit(qc, out_state=out_state)


def v_mul_mps(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    mps_vec: QiskitMPS,
    *,
    trunc_thr: Optional[float] = _NO_TRUNCATION_THR,
) -> QiskitMPS:
    """
    Multiplies circuit matrix by the right-hand side vector: ``out = V @ vec``.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of ansatz parametric circuit.
        mps_vec: right-hand side vector in Qiskit MPS format.
        trunc_thr: truncation threshold in MPS representation.

    Returns:
        out = V @ vec.
    """
    qc = ansatz_to_qcircuit(circ, thetas)
    return qcircuit_mul_mps(qc, mps_vec, trunc_thr=trunc_thr)


def v_dagger_mul_mps(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    mps_vec: QiskitMPS,
    *,
    trunc_thr: Optional[float] = _NO_TRUNCATION_THR,
) -> QiskitMPS:
    """
    Multiplies conjugate-transposed circuit matrix by right-hand side vector:
    ``out = V.H @ vec``. The gates are applied to the vector in inverse order
    because conjugate-transposed matrix is flipped over.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of ansatz parametric circuit.
        mps_vec: right-hand side vector in Qiskit MPS format.
        trunc_thr: truncation threshold in MPS representation.

    Returns:
        out = V.H @ vec.
    """
    qc = ansatz_to_qcircuit(circ, thetas).inverse()  # V.H
    return qcircuit_mul_mps(qc, mps_vec, trunc_thr=trunc_thr)
