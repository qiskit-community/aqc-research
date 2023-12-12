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
Core utilities and classes for time evolution based on Trotterized ansatz.

**Note**, currently we ignore the global phase in Trotter circuit construction
because it makes a mess in gradient computation. Strictly speaking, there must
be a global phase factor ``exp(1j * trotter_global_phase(..))`` evaluated by
the function ``trotter_global_phase()``. One can add it manually, if needed:
``qc.global_phase += trotter_global_phase(..)``. However, as soon as everything
is computed  without phase factor (ansatz and targets), this step can be
postponed until ansatz optimization is done.
"""

from typing import Union, Tuple, Optional
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit
import aqc_research.checking as chk
import aqc_research.utils as helper
from aqc_research.circuit_transform import qcircuit_to_state, qcircuit_to_matrix
import aqc_research.mps_operations as mpsop
from aqc_research.parametric_circuit import (
    ParametricCircuit,
    TrotterAnsatz,
    first_layer_included,
)


class Trotter:
    """
    Class for Trotter evolution of quantum states. Namely, given an input
    initial state and evolution parameters provided in constructor, the
    member functions of this class evolve the initial state forward in time
    using a Trotter circuit.

    By definition, a single "Trotter step" is a full layer of elementary
    2-qubit Trotter blocks applied to every pair of adjacent qubits.
    In case of 2nd order Trotter, additional half-layer is implied at the end.
    """

    def __init__(
        self,
        *,
        num_qubits: int,
        evol_time: float,
        num_steps: int,
        jx:float = 1.0,
        jy:float = 1.0,
        jz:float = 1.0,
        second_order: bool,
    ):
        """
        Args:
            num_qubits: number of qubits.
            evol_time: evolution time.
            num_steps: number of Trotter steps (full layers).
            jx: XX coupling value
            jy = YY coupling value
            jz = ZZ coupling value
            second_order: True, if the 2nd order Trotter is intended.
        """
        assert chk.is_int(num_qubits, num_qubits >= 2)
        assert chk.is_float(evol_time, evol_time > 0)
        assert chk.is_int(num_steps, num_steps >= 1)
        assert all(chk.is_float(j, j > 0) for j in (jx, jy, jz))
        assert isinstance(second_order, bool)

        self._num_qubits = num_qubits
        self._evol_time = evol_time
        self._num_trotter_steps = num_steps
        self._jx = jx
        self._jy = jy
        self._jz = jz
        self._dt = evol_time / float(num_steps)
        self._second_order = second_order

    @property
    def evol_time(self) -> float:
        """Returns the evolution time."""
        return self._evol_time

    @property
    def time_step(self) -> float:
        """Returns the time step in Trotter evolution."""
        return self._dt

    @property
    def num_trotter_steps(self) -> int:
        """Returns the number of steps (full layers) in Trotter algorithm."""
        return self._num_trotter_steps

    def as_vector(self, ini_state: Union[np.ndarray, QuantumCircuit]) -> np.ndarray:
        """
        Time evolution of an initial state by Trotter approximation.

        Args:
            ini_state: a full state vector or a quantum circuit that generates
                       initial state from ``|0>``.

        Returns:
            the state: ``|state> = Trotter |ini_state>``.
        """
        if isinstance(ini_state, np.ndarray):
            assert chk.complex_1d(ini_state)
            qc_ini = QuantumCircuit(helper.num_qubits_from_size(ini_state.size))
        else:
            assert isinstance(ini_state, QuantumCircuit)
            qc_ini = ini_state

        qc = trotter_circuit(
            qc=qc_ini,
            dt=self._dt,
            jx=self._jx,
            jy=self._jy,
            jz=self._jz,
            num_trotter_steps=self._num_trotter_steps,
            second_order=self._second_order,
        )

        if isinstance(ini_state, np.ndarray):
            state = qcircuit_to_matrix(qc) @ ini_state
        else:
            state = qcircuit_to_state(qc)
        return state

    def as_qcircuit(self, ini_state: QuantumCircuit) -> QuantumCircuit:
        """
        Time evolution of an initial state by Trotter approximation. Well, it is
        not a time evolution per se, rather a circuit generation procedure. The
        latter circuit, being applied to zero state ``|0>``, can produce the
        desired result.

        **Note**, the ``ini_state`` circuit will be augmented by the Trotter one.
        Do *not* consider it as an immutable one.

        Args:
            ini_state: quantum circuit that generates initial state from ``|0>``.

        Returns:
            the initial quantum circuit augmented by the Trotter one.
        """
        return trotter_circuit(
            ini_state,
            dt=self._dt,
            jx=self._jx,
            jy=self._jy,
            jz=self._jz,
            num_trotter_steps=self._num_trotter_steps,
            second_order=self._second_order,
        )

    def as_mps(
        self,
        ini_state: QuantumCircuit,
        trunc_thr: float = mpsop.no_truncation_threshold(),
        out_state: Optional[np.ndarray] = None,
    ) -> mpsop.QiskitMPS:
        """
        Time evolution of initial state by Trotter approximation in MPS format.

        Args:
            ini_state: quantum circuit that generates initial state from ``|0>``.
            trunc_thr: truncation threshold in MPS representation.
            out_state: output array for storing state as a normal vector;
                       *note*, this can be very slow and even intractable
                       for a large number of qubits; useful for testing only.

        Returns:
            MPS data generated by Qiskit backend, possibly with state vector,
                if ``out_state`` is provided.
        """
        qc = trotter_circuit(
            ini_state,
            dt=self._dt,
            jx=self._jx,
            jy=self._jy,
            jz=self._jz,
            num_trotter_steps=self._num_trotter_steps,
            second_order=self._second_order,
        )
        return mpsop.mps_from_circuit(qc, trunc_thr=trunc_thr, out_state=out_state)


def make_hamiltonian(num_qubits: int, jx:float=1.0, jy:float=1.0, jz: float=1.0) -> np.ndarray:
    """
    Makes a Hamiltonian matrix. This function is used only for testing to ensure
    generated Trotterized ansatz is consistent with Hamiltonian.

    **Note**, here we use *half-spin* matrices.

    **Remark**: it turns out the bit-ordering does not matter unless
    Hamiltonian is asymmetric, which is *not* this case.

    Args:
        num_qubits: number of qubits.
        jx: XX coupling value
        jy = YY coupling value
        jz = ZZ coupling value

    Returns:
        Hamiltonian matrix.
    """

    def _full_matrix(_s_: np.ndarray, _j_: int) -> np.ndarray:
        """Expands a 2x2 Pauli matrix into a full one."""
        return np.kron(np.kron(np.eye(2**_j_), _s_), np.eye(2 ** (num_qubits - _j_ - 1)))

    def _b2b(_i_: int) -> int:
        """
        Bit-to-bit conversion. See the remark in the parent function doc-string.
        """
        # return num_qubits - 1 - _i  # flip bit-ordering to conform to Qiskit
        return _i_  # do nothing

    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, 0 - 1.0j], [1.0j, 0]], dtype=np.cfloat)
    sigmaz = np.array([[1, 0], [0, -1]])

    sx_ = [_full_matrix(sigmax, j) for j in range(num_qubits)]
    sy_ = [_full_matrix(sigmay, j) for j in range(num_qubits)]
    sz_ = [_full_matrix(sigmaz, j) for j in range(num_qubits)]

    rng = range(num_qubits - 1)
    sx_sx = [np.dot(sx_[_b2b(i)], sx_[_b2b(i + 1)]) for i in rng]
    sy_sy = [np.dot(sy_[_b2b(i)], sy_[_b2b(i + 1)]) for i in rng]
    sz_sz = [np.dot(sz_[_b2b(i)], sz_[_b2b(i + 1)]) for i in rng]

    xterms = np.sum(sx_sx, axis=0)
    yterms = np.sum(sy_sy, axis=0)
    zterms = np.sum(sz_sz, axis=0)

    h = -0.25 * (jx * xterms + jy * yterms + jz * zterms)
    return h


def exact_evolution(
    hamiltonian: np.ndarray,
    ini_state: Union[QuantumCircuit, np.ndarray],
    evol_time: float,
) -> np.ndarray:
    """
    Computes exact state evolution starting from the initial state.
    This function is used only for testing to ensure generated Trotterized
    ansatz is consistent with Hamiltonian.

    **Note**, might be a slow routine, suitable for debugging and testing
    with a moderate number of qubits.

    Args:
        hamiltonian: Hamiltonian for state evolution.
        ini_state: quantum circuit acting on the state ``|0>``
                   to produce an initial one, or a corresponding quantum state.
        evol_time: evolution time.

    Returns:
        final state as the result of time evolution of the initial one.
    """
    assert chk.complex_2d(hamiltonian)
    assert isinstance(ini_state, (QuantumCircuit, np.ndarray))
    assert chk.is_float(evol_time, evol_time > 0)

    if isinstance(ini_state, QuantumCircuit):
        ini_state = qcircuit_to_state(ini_state)
    assert chk.complex_1d(ini_state)
    assert hamiltonian.shape == (ini_state.size, ini_state.size)

    e_h = expm((-1.0j * evol_time) * hamiltonian)
    exact_state = np.matmul(e_h, ini_state)
    return exact_state


def trotter_alphas(dt: float, jx: float, jy:float, jz:float) -> np.ndarray:
    """
    Computes 3 angular parameters (``alphas``) of a Trotter building block.

    Args:
        dt: time step in Trotter algorithm.
        jx: XX coupling value
        jy = YY coupling value
        jz = ZZ coupling value

    Returns:
        angular parameters of Trotter building block.
    """
    assert chk.is_float(dt, dt > 0)
    assert all(chk.is_float(j, j > 0) for j in (jx, jy, jz))

    return np.asarray([np.pi / 2 - 0.5 * jz * dt, 0.5 * dt * jx - np.pi / 2, np.pi / 2 - 0.5 * dt * jy])


def trotter_global_phase(num_qubits: int, num_steps: int, second_order: bool) -> float:
    """
    Returns global phase of a Trotter circuit. Note, after Trotter, the global
    phase of a qcircuit instance should be incremented as follows:
    ``qcircuit.global_phase += exp(1j * (global phase))``.

    Args:
        num_qubits: number of qubits.
        num_steps: number of Trotter steps.
        second_order: True, if the 2nd order Trotter is intended.

    Returns:
        global phase of Trotter circuit.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert chk.is_int(num_steps, num_steps >= 1)
    assert isinstance(second_order, bool)

    quarter_pi = 0.25 * np.pi
    phs = quarter_pi * (num_qubits - 1) * num_steps
    if second_order:
        if num_qubits % 2 == 0:  # even
            # return ph + quarter_pi * (num_qubits // 2)
            return phs + quarter_pi * num_qubits
        else:  # odd
            # return ph + quarter_pi * ((num_qubits - 1) // 2)
            return phs + quarter_pi * (num_qubits - 1)
    else:
        return phs


def trotter_circuit(
    qc: QuantumCircuit,
    *,
    dt: float,
    jx=1.0,
    jy=1.0,
    jz=1.0,
    num_trotter_steps: int,
    second_order: bool,
) -> QuantumCircuit:
    """
    Generates a 1st or 2nd order Trotter circuit and adds it to the input one.
    By definition, a single "Trotter step" is a full layer of elementary.
    2-qubit Trotter blocks applied to every pair of adjacent qubits.
    The parameter ``num_trotter_steps`` defines the number of layers in the
    circuit. The parameter ``dt`` characterizes the evolution time per step
    (layer). The total evolution time is equal to ``dt * num_trotter_steps``.

    **Note**, 2nd order Trotter circuit comprises additional half-layer at the end.

    **Note**, currently we ignore the global phase, see the remark at the
    beginning of this script.

    Args:
        qc: quantum circuit to be augmented by the Trotter one.
        dt: evolution time per step (layer) in Trotter algorithm.
        jx: XX coupling value
        jy = YY coupling value
        jz = ZZ coupling value
        num_trotter_steps: number of Trotter steps (layers).
        second_order: True, if the 2nd order Trotter is intended.

    Returns:
        quantum circuit augmented by the Trotter one.
    """
    assert isinstance(qc, QuantumCircuit) and qc.num_qubits > 0
    assert chk.is_int(num_trotter_steps, num_trotter_steps > 0)

    def _trotter_block(k: int, params: np.ndarray):
        qc.rz(-np.pi / 2, k + 1)
        qc.cx(k + 1, k)
        qc.rz(params[0], k)
        qc.ry(params[1], k + 1)
        qc.cx(k, k + 1)
        qc.ry(params[2], k + 1)
        qc.cx(k + 1, k)
        qc.rz(np.pi / 2, k)

    # Compute Trotter parameters. In case of 2nd order, the first and the trail
    # half-layers should be initialized differently ("betas").
    alphas = trotter_alphas(dt, jx, jy, jz)
    betas = trotter_alphas(dt * 0.5, jx, jy, jz)  # dt/2 (!) in first/last half-layers

    # Build the main part of the 1st or 2nd order Trotter circuit.
    for j in range(num_trotter_steps):
        for q in range(0, qc.num_qubits - 1, 2):  # 1st half of a layer
            _trotter_block(q, betas if second_order and j == 0 else alphas)
        for q in range(1, qc.num_qubits - 1, 2):  # 2nd half of a layer
            _trotter_block(q, alphas)

    # For 2nd order Trotter, we add an extra half-layer identical to the front one.
    if second_order:
        for q in range(0, qc.num_qubits - 1, 2):
            _trotter_block(q, betas)

    return qc


def identity_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Returns the identity (empty) quantum circuit.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    return QuantumCircuit(num_qubits)


def neel_init_state(num_qubits: int) -> QuantumCircuit:
    """
    Returns quantum circuit that produces the state ``|101010...>``
    (Neel state) of alternating units from the state ``|0>``.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    qc = QuantumCircuit(num_qubits)
    for k in range(0, num_qubits, 2):
        qc.x(k)
    return qc


def half_zero_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Returns quantum circuit that produces the state ``|00...0011...11>``
    of half-zero/half-unit bits from the state ``|0>``.
    """
    assert chk.is_int(num_qubits, num_qubits >= 2)
    qc = QuantumCircuit(num_qubits)
    for k in range(num_qubits // 2, num_qubits):
        qc.x(k)
    return qc


def fidelity(
    state1: Union[mpsop.QiskitMPS, np.ndarray],
    state2: Union[mpsop.QiskitMPS, np.ndarray],
) -> float:
    """Computes fidelity between two states, which must have the same type."""
    if isinstance(state1, np.ndarray) and isinstance(state2, np.ndarray):
        assert chk.complex_1d(state1) and chk.complex_1d(state2)
        return float(np.abs(np.vdot(state1, state2)) ** 2)
    else:
        return float(np.abs(mpsop.mps_dot(state1, state2)) ** 2)


def state_difference(state1: np.ndarray, state2: np.ndarray) -> float:
    """Computes norm of state difference. **Note**, phase factor can crucial."""
    assert chk.complex_1d(state1) and chk.complex_1d(state2)
    return float(np.linalg.norm(state1 - state2))


def slice2q(
    circ: ParametricCircuit,
    vec: np.ndarray,
    *,
    layer_range: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Returns a slice of input vector's entries pertaining to the range of
    layers specified. Actually, a view of the vector is returned.

    **Note**, here ``layer`` is a collection of (num_qubit - 1) triplets of
    unit-blocks. Triplet structure of ansatz resembles (but not coincides with)
    the Trotter circuit and has 12 parameters.

    Args:
         circ: parametrized ansatz circuit.
         vec: vector of parameters or gradients to be sliced.
         layer_range: range of layers to get a slice of vector entries for;
                      entire range is implied for the None value.

    Returns:
        (1) 3D array of reshaped entries of the input vector, the 1st index
        enumerates selected layers in the range, the 2nd index enumerates 2-qubit
        triplets of blocks (Trotter units), the 3rd index enumerates 1-qubit gates
        in a triplet (and corresponding entries in ``vec``).
        (2) range of layers; validated to the full range in case of None
        input value.
    """
    if not isinstance(circ, TrotterAnsatz):
        raise ValueError("expects Trotterized ansatz")
    assert isinstance(vec, np.ndarray) and vec.shape == (circ.num_thetas,)

    num_layers = circ.num_layers
    layer_range = (0, num_layers) if layer_range is None else layer_range

    assert chk.is_tuple(layer_range, len(layer_range) == 2)
    assert num_layers * circ.bpl == circ.num_blocks
    assert 0 <= layer_range[0] < layer_range[1] <= num_layers

    # Get a sub-set of layers, each layer consists of n-1 triplets of CX-blocks
    # with 12 (3 * 4) angular parameters in every triplet.
    vec2q = circ.subset2q(vec).reshape((num_layers, circ.num_qubits - 1, 12))
    vec2q = vec2q[layer_range[0] : layer_range[1]]
    assert np.shares_memory(vec2q.ravel(), vec)  # "vec2q" is a view of "vec"
    return vec2q, layer_range


def init_ansatz_to_trotter(
    circ: ParametricCircuit,
    thetas: np.ndarray,
    *,
    evol_time: float,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    layer_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Modifies the angular parameter ``thetas``, within specified range of layers,
    in such way that ansatz becomes equivalent the Trotter circuit. This function
    is used to generate the best possible initial vector of angular parameters
    for optimization.

    Args:
        circ: parametric circuit associated with this objective.
        thetas: angular parameters of the circuit.
        evol_time: evolution time; unit-block layers within ``layer_range``
                   should model state evolution for that time.
        jx: XX coupling value
        jy = YY coupling value
        jz = ZZ coupling value
        layer_range: a couple of indices ``[from, to)`` that defines a range of
                     unit-block layers to be initialized; None value implies
                     a full range.

    Returns:
        vector of angular parameters ``thetas`` initialized to reproduce
        Trotter circuit.
    """
    th2q, layer_range = slice2q(circ, thetas, layer_range=layer_range)
    delta_t = evol_time / float(layer_range[1] - layer_range[0])
    alphas = trotter_alphas(dt=delta_t, jx=jx, jy=jy, jz=jz)
    assert chk.float_1d(alphas, alphas.size == 3)
    assert isinstance(circ, TrotterAnsatz)
    layer_0 = first_layer_included(circ, layer_range)

    # If the first layer of unit-blocks is included, we set the front layer
    # of 1-qubit gates to zero. Do NOT be confused: "front layer" of 1-qubit
    # gates and the "first layer" of 2-qubit unit-blocks are different notions.
    if layer_0:
        circ.subset1q(thetas).fill(0)

    # Most of angular parameters are equal zero except 3 ones per block triplet.
    th2q.fill(0)
    th2q[:, :, 5] = alphas[0]
    th2q[:, :, 0] = alphas[1]
    th2q[:, :, 6] = alphas[2]

    # In case of the 2nd order Trotter and if the first layer is included in the
    # requested range, we initialize differently the front and trail half-layers.
    # Recall, the trailing half-layer takes exactly the same parameters as the
    # first one, although it does not present explicitly in TrotterAnsatz.
    if circ.is_second_order and layer_0:
        alphas = trotter_alphas(dt=delta_t * 0.5, jx=jx, jy=jy, jz=jz)  # dt/2 (!)
        half = circ.half_layer_num_blocks // 3  # half of triplets in layer_0
        assert 3 * half == circ.half_layer_num_blocks  # divisible
        th2q[0, 0:half, 5] = alphas[0]
        th2q[0, 0:half, 0] = alphas[1]
        th2q[0, 0:half, 6] = alphas[2]

    return thetas
