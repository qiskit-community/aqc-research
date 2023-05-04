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
Base class for computation of objective function and its gradient designed
for state preparation with local Hilbert-Schmidt objective. It is assumed
by default that all flip-terms in objective function have the same weight.
The script also includes auxiliary classes that facilitate state preparation.
"""

from abc import ABC, abstractmethod
import itertools
import functools
from typing import Optional, Tuple, Union, List, Callable
import numpy as np
from qiskit import QuantumCircuit
import aqc_research.checking as chk
from aqc_research.parametric_circuit import ParametricCircuit, TrotterAnsatz
import aqc_research.utils as utl
import aqc_research.core_operations as cop
from aqc_research.circuit_transform import qcircuit_to_state
from aqc_research.mps_operations import QiskitMPS, mps_from_circuit, mps_dot, check_mps
from aqc_research.optimizer import TimeoutChecker, EarlyStopper

_logger = utl.create_logger(__file__)


# -----------------------------------------------------------------------------
# Handlers of flip-states and zero-state that make up a subspace for optimization.
# -----------------------------------------------------------------------------


class ThinStateHandler:
    """
    Class initializes the flip-states and zero one ``|0>``.
    The main point of this implementation: the states ``|0>`` and ``X_i @ |0>``
    are **not** explicitly cached. Instead, they are constructed upon demand.
    This strategy reduces memory footprint.
    **Note**, construction of full state vectors is suitable for simulations
    with a moderate number of qubits.
    """

    def __init__(self, num_qubits: int, max_flips: int, verbose: bool = False):
        """
        Args:
            num_qubits: number of qubits.
            max_flips: max. number of bit flips per state.
            verbose: enables verbose status messages.
        """
        assert chk.is_int(num_qubits, num_qubits >= 2)
        assert chk.is_int(max_flips, 0 <= max_flips <= num_qubits)
        assert chk.is_bool(verbose)
        if verbose:
            _logger.info("State handler: %s", self.__class__.__name__)

        qiskit_convention = bool(cop.bit2bit_transform(num_qubits, 0) != 0)
        dim = 2**num_qubits
        comb_labels, num_states = self._generate_combinations(num_qubits, max_flips)
        self._comb_labels = comb_labels
        self._num_qubits = num_qubits
        self._max_flips = max_flips

        # Placeholder for a (temporary) flip-state or |0>:
        self._state = np.zeros(dim, dtype=np.cfloat)

        # Indices of (a single) non-zero entry in every state:
        self._state_idx = np.zeros(num_states, dtype=np.int64)

        # Pre-compute the states |0>, X_j@|0>, X_i@X_j@|0>, etc.
        self._state_idx[0] = 0
        count = 1
        for flips in range(max_flips):
            for j in range(len(comb_labels[flips])):
                index = np.int64(0)
                for i in range(flips + 1):
                    k = comb_labels[flips][j][i]
                    if qiskit_convention:
                        index ^= 1 << k
                    else:
                        index ^= 1 << (num_qubits - 1 - k)  # change bit ordering
                assert 0 <= index < dim
                self._state_idx[count] = index  # index of non-zero entry
                count += 1
        assert count == num_states

    def init_state(self, state_no: int) -> np.ndarray:
        """
        Initializes a state vector in the form |0>, X_i@|0>, X_i@X_j@|0>, etc.
        **Note**, each state is a vector of all zeros except one unit element.
        **Note**, one can safely modify the content of the output array because
        it will be completely re-initialized upon next call.

        Args:
            state_no: index of a state in question.

        Returns:
            the internal array initialized to requested state.
        """
        assert chk.is_int(state_no, 0 <= state_no < self.num_states)

        self._state.fill(0)
        self._state[self._state_idx[state_no]] = 1
        return self._state

    @property
    def state0(self) -> np.ndarray:
        """Returns the first state."""
        return self.init_state(0)

    def init_composite_state_no_zero(self, coefs: np.ndarray) -> np.ndarray:
        """
        Initializes a state vector as a linear combination of *flip-states*.
        **Note**, the state ``|0>`` is *not* included in the linear combination.
        **Note**, each state is a vector of all zeros except one unit element.
        **Note**, one can safely modify the content of the output array because
        it will be completely re-initialized upon next call.
        **Note**, ``coefs`` should define a linear combination of ket-vectors:
        ``|composite state> = sum_i coefs_i |state_i>``.

        Args:
            coefs: coefficients of the linear combination of *flip-states*.

        Returns:
            the internal array initialized to requested composite state.
        """
        assert chk.complex_or_float_1d(coefs, coefs.size == self.num_states - 1)
        assert abs(np.linalg.norm(coefs) - 1) < np.sqrt(np.finfo(np.float64).eps)

        self._state.fill(0)
        self._state[self._state_idx[1:]] = coefs
        return self._state

    def init_composite_state(self, coefs: np.ndarray) -> np.ndarray:
        """
        Initializes a state vector as a linear combination of *all* states.
        **Note**, each state is a vector of all zeros except one unit element.
        **Note**, one can safely modify the content of the output array because
        it will be completely re-initialized upon next call.
        **Note**, ``coefs`` should define a linear combination of ket-vectors:
        ``|composite state> = sum_i coefs_i |state_i>``.

        Args:
            coefs: coefficients of the linear combination of *all* states.

        Returns:
            the internal array initialized to requested composite state.
        """
        assert chk.complex_or_float_1d(coefs, coefs.size == self.num_states)
        assert abs(np.linalg.norm(coefs) - 1) < np.sqrt(np.finfo(np.float64).eps)

        self._state.fill(0)
        self._state[self._state_idx] = coefs
        return self._state

    def state_dot_vector(self, state_no: int, vec: np.ndarray) -> np.cfloat:
        """
        Computes <state|vector>, where state is a vector of all zeros except
        one unit element. Because state has a single non-zero element, we just
        pick up and return the corresponding element of the input vector.
        """
        assert chk.is_int(state_no, 0 <= state_no < self.num_states)
        assert chk.complex_1d(vec, vec.size == 2**self._num_qubits)

        return vec[self._state_idx[state_no]]

    def composite_state_dot_vector_no_zero(self, coefs: np.ndarray, vec: np.ndarray) -> np.cfloat:
        """
        Computes ``<composite state|vector>``, where "composite state" is a vector
        of linear combination of *flip-states*. Composite state is very sparse,
        so we can compute the dot product over few non-zero entries.
        **Note**, the state ``|0>`` is not included in the linear combination.
        **Note**, ``coefs`` should define a linear combination of ket-vectors:
        ``|composite state> = sum_i coefs_i |state_i>``.

        Args:
            coefs: coefficients of the linear combination of *flip-states*.
            vec: right-hand side vector.

        Returns:
            complex dot product value.
        """
        assert chk.complex_or_float_1d(coefs, coefs.size == self.num_states - 1)
        assert abs(np.linalg.norm(coefs) - 1) < np.sqrt(np.finfo(np.float64).eps)
        assert chk.complex_1d(vec, vec.size == 2**self._num_qubits)

        return np.cfloat(np.vdot(coefs, vec[self._state_idx[1:]]))

    def composite_state_dot_vector(self, coefs: np.ndarray, vec: np.ndarray) -> np.cfloat:
        """
        Computes ``<composite state|vector>``, where "composite state" is
        a vector of linear combination of *all* states. Composite state is very
        sparse, so we can compute the dot product over few non-zero entries.
        **Note**, ``coefs`` should define a linear combination of ket-vectors:
        ``|composite state> = sum_i coefs_i |state_i>``.

        Args:
            coefs: coefficients of the linear combination of *all* states.
            vec: right-hand side vector.

        Returns:
            complex dot product value.
        """
        assert chk.complex_or_float_1d(coefs, coefs.size == self.num_states)
        assert abs(np.linalg.norm(coefs) - 1) < np.sqrt(np.finfo(np.float64).eps)
        assert chk.complex_1d(vec, vec.size == 2**self._num_qubits)

        return np.cfloat(np.vdot(coefs, vec[self._state_idx]))

    @property
    def num_states(self) -> int:
        """
        Returns the number of flip-states including |0>.
        """
        return self._state_idx.size

    @property
    def flip_qubit_positions(self) -> List[List[Tuple]]:
        """
        Returns all possible combinations of 1, 2, 3, etc. X-gates placements.
        Those gates flip the corresponding qubits. Function is useful for testing.
        """
        return self._comb_labels

    @staticmethod
    def _generate_combinations(num_qubits: int, max_flips: int) -> Tuple[list, int]:
        """
        Generates all possible subsets of indices pertaining to combinations
        of ``X`` gates: ``{X_i}``, ``{X_i X_j}``, ``{X_i X_j X_k}``, etc.

        Args:
            num_qubits: number of qubits.
            max_flips: max. number of bit flips per state.

        Returns:
            index combinations, number of states including ``|0>``.
        """
        # Find all possible combinations of 1, 2, 3, etc. X-gates defined.
        s = list(range(num_qubits))
        comb_labels = [[] for _ in range(max_flips)]
        for flip in range(1, max_flips + 1):
            for subset in itertools.combinations(s, flip):
                comb_labels[flip - 1].append(subset)

        # Compute the number of combinations including (!) the bare |0>.
        num_states = functools.reduce(lambda n, a: n + len(a), comb_labels, 1)
        return comb_labels, num_states


class GenericStateHandler:
    """
    Class initializes the flip-states and zero one ``|0>``.
    In this implementation all the states ``S @ |0>`` and ``S @ X_i @ |0>`` are
    explicitly constructed and cached, where ``S`` is some state preparation
    circuit (possibly just an identity one). This strategy leads to additional
    memory consumption but enables more generic initial and flipped states.
    **Note**, construction of full state vectors is suitable for simulations
    with a moderate number of qubits.
    """

    def __init__(
        self,
        num_qubits: int,
        max_flips: int,
        state_prep_func: Optional[Callable[[int], QuantumCircuit]] = None,
        verbose: bool = False,
    ):
        """
        Args:
            num_qubits: number of qubits.
            max_flips: max. number of bit flips per state; currently expects
                       the value 0 or 1 in order to save memory.
            state_prep_func: function that creates a quantum circuit to be applied
                             after initial flip-state or zero state; None implies
                             an empty circuit.
            verbose: enables verbose status messages.
        """
        assert chk.is_int(num_qubits, num_qubits >= 2)
        assert chk.is_int(max_flips, 0 <= max_flips <= num_qubits)
        assert callable(state_prep_func)
        if verbose:
            _logger.info("State handler: %s", self.__class__.__name__)

        if max_flips > 1:
            raise ValueError("expects 'max_flips <= 1' to save memory")

        num_states = num_qubits + 1
        self._states = np.zeros((num_states, 2**num_qubits), dtype=np.cfloat)
        for i in range(num_states):
            qc = QuantumCircuit(num_qubits)
            if i > 0:
                qc.x(i - 1)  # single flip is only supported so far
            if state_prep_func is not None:
                qc = qc.compose(state_prep_func(num_qubits))  # S applied after flip
            np.copyto(self._states[i], qcircuit_to_state(qc))

    def init_state(self, state_no: int) -> np.ndarray:
        """
        Returns one of the flip-states or zero state upon request.
        The output vector must *not* be modified.
        """
        assert chk.is_int(state_no, 0 <= state_no < self.num_states)
        return self._states[state_no]

    def state_dot_vector(self, state_no: int, vec: np.ndarray) -> np.cfloat:
        """Computes <state|vector>."""
        assert chk.is_int(state_no, 0 <= state_no < self.num_states)
        return np.cfloat(np.vdot(self._states[state_no], vec))

    @property
    def state0(self) -> np.ndarray:
        """Returns the first state."""
        return self._states[0]

    @property
    def num_states(self) -> int:
        """Returns the number of flip-states plus non-flipped one."""
        return self._states.shape[0]

    def init_composite_state_no_zero(self, _: np.ndarray) -> np.ndarray:
        """Interface stub function. Not implemented."""
        raise NotImplementedError()

    def init_composite_state(self, _: np.ndarray) -> np.ndarray:
        """Interface stub function. Not implemented."""
        raise NotImplementedError()

    def composite_state_dot_vector_no_zero(self, _: np.ndarray, __: np.ndarray) -> np.cfloat:
        """Interface stub function. Not implemented."""
        raise NotImplementedError()

    def composite_state_dot_vector(self, _: np.ndarray, __: np.ndarray) -> np.cfloat:
        """Interface stub function. Not implemented."""
        raise NotImplementedError()


class MpsStateHandler:
    """
    Class initializes the flip-states and zero one ``|0>`` in MPS representation.
    In this implementation all the states ``S @ |0>`` and ``S @ X_i @ |0>`` are
    explicitly constructed and cached, where ``S`` is some state preparation
    circuit (possibly just an identity one). Here we use MPS representation of
    state vectors which can handle a large number of qubits.
    """

    def __init__(
        self,
        num_qubits: int,
        max_flips: int,
        state_prep_func: Optional[Callable[[int], QuantumCircuit]] = None,
        verbose: bool = False,
    ):
        """
        Args:
            num_qubits: number of qubits.
            max_flips: max. number of bit flips per state; currently expects
                       the value 0 or 1 in order to save memory and comp. time.
            state_prep_func: function that creates a quantum circuit to be applied
                             after initial flip-state or zero state; None implies
                             an empty circuit.
            verbose: enables verbose status messages.
        """
        assert chk.is_int(num_qubits, num_qubits >= 2)
        assert chk.is_int(max_flips, 0 <= max_flips <= num_qubits)
        assert callable(state_prep_func)
        if verbose:
            _logger.info("State handler: %s", self.__class__.__name__)

        if max_flips > 1:
            raise ValueError("expects 'max_flips <= 1' to save memory & time")

        # Use default truncation threshold. States are assumed low-entangled.
        num_states = num_qubits + 1
        self._states: List[QiskitMPS] = list([])
        for i in range(num_states):
            qc = QuantumCircuit(num_qubits)
            if i > 0:
                qc.x(i - 1)  # single flip is only supported so far
            if state_prep_func is not None:
                qc = qc.compose(state_prep_func(num_qubits))  # S applied after flip
            self._states.append(mps_from_circuit(qc))
        assert len(self._states) == num_states

    def init_state(self, state_no: int) -> QiskitMPS:
        """
        Returns one of the flip-states or zero state upon request.
        The output vector must *not* be modified.
        """
        assert chk.is_int(state_no, 0 <= state_no < self.num_states)
        return self._states[state_no]

    def state_dot_vector(self, state_no: int, vec: QiskitMPS) -> np.cfloat:
        """Computes <state|vector>."""
        assert chk.is_int(state_no, 0 <= state_no < self.num_states)
        return np.cfloat(mps_dot(self._states[state_no], vec))

    @property
    def state0(self) -> QiskitMPS:
        """Returns the first state."""
        return self._states[0]

    @property
    def num_states(self) -> int:
        """Returns the number of flip-states plus non-flipped one."""
        return len(self._states)

    def init_composite_state_no_zero(self, _: np.ndarray) -> np.ndarray:
        """Interface stub function. Not implemented."""
        raise NotImplementedError()

    def init_composite_state(self, _: np.ndarray) -> np.ndarray:
        """Interface stub function. Not implemented."""
        raise NotImplementedError()

    def composite_state_dot_vector_no_zero(self, _: np.ndarray, __: np.ndarray) -> np.cfloat:
        """Interface stub function. Not implemented."""
        raise NotImplementedError()

    def composite_state_dot_vector(self, _: np.ndarray, __: np.ndarray) -> np.cfloat:
        """Interface stub function. Not implemented."""
        raise NotImplementedError()


# -----------------------------------------------------------------------------
# Convenience classes that facilitate the optimization process.
# -----------------------------------------------------------------------------


class SpService:
    """
    Class handles convenience services such as checking for early termination,
    collecting the statistics, incrementing the iteration counters, etc.
    """

    def __init__(
        self,
        user_parameters: dict,
        circuit: ParametricCircuit,
        num_states: int,
        verbose: bool = False,
    ):
        """
        Args:
            user_parameters: user supplied settings.
            circuit: instance of a parametric circuit.
            num_states: number of states that make up terms in objective function.
            verbose: enables verbose status messages.
        """
        super().__init__()
        assert chk.is_dict(user_parameters)
        assert isinstance(circuit, ParametricCircuit)
        assert chk.is_int(num_states, num_states >= 1)
        assert chk.is_bool(verbose)
        if verbose:
            _logger.info("State preparation service: %s", self.__class__.__name__)

        self._params = user_parameters
        self._circuit = circuit
        self._num_states = num_states
        self._verbose = verbose

        self._num_fun_ev = int(0)
        self._num_grad_ev = int(0)
        self._stats = dict({})
        self._timeout_checker: Union[TimeoutChecker, None] = None
        self._early_stopper: Union[EarlyStopper, None] = None

        if user_parameters["enable_optim_stats"]:
            self._stats = {
                "hs2": np.empty((0, num_states), dtype=np.float16),
                "weight": np.empty(0, dtype=np.float16),
                "fobj": np.empty(0, dtype=np.float32),
                "grad": np.empty(0, dtype=np.float32),
                "num_fun_ev": int(0),
                "num_grad_ev": int(0),
            }

    def set_status_trackers(
        self,
        timeout: Optional[TimeoutChecker] = None,
        stopper: Optional[EarlyStopper] = None,
    ):
        """
        Initializes reference(s) to the object(s) for early interruption
        the optimization process upon certain condition(s).

        Args:
            timeout: instance of timeout checker or None.
            stopper: instance of early stopper or None.
        """
        self._timeout_checker = timeout
        self._early_stopper = stopper

    @property
    def statistics(self) -> dict:
        """
        Returns:
            optimization statistics, if available, or empty dictionary.
        """
        return self._stats

    def _on_stop(self, fobj: float, thetas: np.ndarray) -> dict:
        """
        Invoked before StopIteration or Timeout is fired.
        **Note**: circuit  structure can in principle change over time,
        therefore we always save ``blocks``.

        Args:
            fobj: the best objective function value.
            thetas: the best angular parameters as arguments.

        Returns:
            dictionary of optimization results.
        """
        if self._verbose:
            print()
            _logger.warning("Early stopping of the optimization process")
        return {
            "cost": fobj,
            "num_fun_ev": self._num_fun_ev,
            "num_grad_ev": self._num_grad_ev,
            "num_iters": self._num_grad_ev,
            "thetas": thetas.copy(),
            "blocks": self._circuit.blocks.copy(),
        }

    def on_begin_gradient(self, fobj: float, thetas: np.ndarray, fidelity: Optional[float] = None):
        """
        Checks if optimization process should be terminated. Invoked at the
        beginning of gradient computation.

        Args:
            fobj: current value of objective function.
            thetas: current angular parameters we optimize for.
            fidelity: fidelity between the current and target states.
        """
        if self._timeout_checker:
            self._timeout_checker.check(fobj, thetas, self._on_stop)
        if self._early_stopper:
            self._early_stopper.check(
                fobj=fobj,
                fidelity=fidelity,
                thetas=thetas,
                iter_no=self._num_grad_ev,
                on_stop=self._on_stop,
            )

    def on_end_gradient(
        self,
        fobj: float,
        fidelity: float,
        grad: np.ndarray,
        hs2: np.ndarray,
        weight: float,
    ):
        """
        Prints progress, updates statistics, increments counter(s).
        Invoked at the end of gradient computation.

        Args:
            fobj: current value of objective function.
            fidelity: current fidelity value, negative value will be ignored.
            grad: current vector of gradients.
            hs2: 1D array of Hilbert-Schmidt products of all the states.
            weight: common scale of all the flip-terms.
        """
        assert chk.is_float(fobj)
        assert chk.float_1d(grad)
        assert chk.float_1d(hs2, hs2.size == self._num_states)
        assert chk.is_float(weight, weight >= 0)

        # Increment counter of the number of gradient evaluations.
        self._num_grad_ev += 1

        # Accumulate statistics.
        verbose = self._params["verbose"]
        if self._params["enable_optim_stats"]:
            grad_norm = np.linalg.norm(grad)
            sts = self._stats
            sts["hs2"] = np.insert(sts["hs2"], sts["hs2"].shape[0], hs2, axis=0)
            sts["weight"] = np.append(sts["weight"], np.float16(weight))
            sts["fobj"] = np.append(sts["fobj"], np.float32(fobj))
            sts["grad"] = np.append(sts["grad"], np.float32(grad_norm))
            sts["num_fun_ev"] = self._num_fun_ev
            sts["num_grad_ev"] = self._num_grad_ev
            sts["num_iters"] = self._num_grad_ev

        # Print progress. More information is printed out in case a single job.
        if self._num_grad_ev % max(1, self._params["maxiter"] // 50) == 0:
            if verbose > 0 and self._params["num_simulations"] == 1:
                fid_str = f", fidelity: {fidelity:0.6f}" if fidelity >= 0 else ""
                _logger.info("fobj: %0.6f %s", fobj, fid_str)
            else:
                print(".", end="", flush=True)

    def on_end_objective(self):
        """
        Invoked at the end of computation of an objective function.
        """
        self._num_fun_ev += 1

    def on_epoch_end(self):
        """
        Marks the end of optimization stage or epoch.
        """
        if self._verbose:
            _logger.warning("End of optimization epoch")
        # Append NaN to statistics to make a visual gap in graphical plots.
        if len(self._stats) > 0:
            sts = self._stats
            sts["hs2"] = np.insert(sts["hs2"], sts["hs2"].shape[0], np.nan, axis=0)
            sts["weight"] = np.append(sts["weight"], np.float16(np.nan))
            sts["fobj"] = np.append(sts["fobj"], np.float32(np.nan))
            sts["grad"] = np.append(sts["grad"], np.float32(np.nan))


# -----------------------------------------------------------------------------
# Classes related to construction of an objective function.
# -----------------------------------------------------------------------------


class SpLHSObjectiveBase(ABC):
    """
    Base class for implementing of objective function and its gradient designed
    for state preparation with local Hilbert-Schmidt objective. It is assumed
    by default that all flip-terms in objective function have the same weight.
    """

    def __init__(
        self,
        user_parameters: dict,
        circuit: ParametricCircuit,
        use_mps: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            user_parameters: user supplied parameters.
            circuit: instance of a parametric circuit.
            use_mps: non-zero fi objective's implementation relies on MPS
                     rather than classical vector states.
            verbose: enables verbose status messages.
        """
        super().__init__()
        assert isinstance(user_parameters, dict)
        assert isinstance(circuit, ParametricCircuit)
        assert isinstance(use_mps, bool)
        assert chk.is_bool(verbose)
        if verbose:
            _logger.info("Objective: %s", self.__class__.__name__)
            if isinstance(circuit, TrotterAnsatz):
                _logger.info("Trotterized ansatz is being used in objective")

        self._params = user_parameters
        self._circuit = circuit
        self._target: Union[np.ndarray, QiskitMPS, None] = None
        self._last_thetas = np.empty(0)  # the latest thetas passed in objective

        self._use_mps = use_mps  # non-zero if MPS states are used
        self._verbose = verbose
        self._print_grad_warning = True  # enables warning: grad. before objective

        # Intermediate storage and data cached in objective(..) function:
        self._vh_target: Union[np.ndarray, QiskitMPS, None] = None
        self._workspace: Union[np.ndarray, None] = None
        if not use_mps:
            self._vh_target = np.zeros(circuit.dimension, dtype=np.cfloat)
            self._workspace = np.zeros((3, circuit.dimension), dtype=np.cfloat)

        # State handler:
        num_qubits = user_parameters["num_qubits"]
        max_flips = user_parameters["max_flips"]
        state_prep_func = user_parameters.get("state_prep_func", None)
        if use_mps:
            self._state_handler = MpsStateHandler(num_qubits, max_flips, state_prep_func, verbose)
            self._num_states = num_qubits + 1
            if max_flips != 1:
                raise ValueError("expects max_flips=1 in case of using MPS")
        else:
            if state_prep_func is None:
                self._state_handler = ThinStateHandler(num_qubits, max_flips, verbose)
            else:
                self._state_handler = GenericStateHandler(
                    num_qubits, max_flips, state_prep_func, verbose
                )
            self._num_states = self._state_handler.num_states

        # Infrastructure service:
        self._service = SpService(user_parameters, circuit, self._num_states, verbose=verbose)

        # Hilbert-Schmidt products squared |<state|V.H|target>|^2:
        self._hs2 = np.zeros(self._num_states)

        # Current objective function value and common weight of the flip terms:
        self._fobj = 1.0
        self._weight = 1.0

    def _store_latest_thetas(self, thetas: np.ndarray):
        """
        Stores the latest vector of angular circuit parameters.
        """
        if self._last_thetas.size == 0:
            self._last_thetas = thetas.copy()
        else:
            np.copyto(self._last_thetas, thetas)

    def _calc_objective_before_gradient(self, thetas: np.ndarray):
        """
        Some optimizers can compute gradient before the objective function or
        not even using the objective() at all (ADAM). However, we need to be
        sure that objective was computed before the gradient with exactly the
        same variables (here, angular parameters "theta"). If thetas are the
        same as the ones used for objective value calculation before calling
        this function, then we re-use the computations, otherwise we have to
        re-compute the objective.

        Args:
            thetas: current angular parameters.
        """
        tol = float(np.sqrt(np.finfo(np.float64).eps))
        last = self._last_thetas
        if last.size == 0 or not np.allclose(thetas, last, atol=tol, rtol=tol):
            self.objective(thetas)
            if self._verbose and self._print_grad_warning:
                _logger.warning("enforcing computation of the objective before the gradient")
                self._print_grad_warning = False  # print just once

    @abstractmethod
    def objective(self, thetas: np.ndarray) -> float:
        """
        Computes a value of the objective function given a vector of angular
        parameter values.

        Args:
            thetas: a vector of angular parameter values for the optimization problem.

        Returns:
            a float value of the objective function.
        """
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """
        Computes a gradient with respect to angular parameters given a vector
        of parameter values.

        Args:
            thetas: a vector of angular parameter values for the optimization problem.

        Returns:
            a real array of gradient values.
        """
        raise NotImplementedError()

    def set_status_trackers(
        self,
        timeout: Optional[TimeoutChecker] = None,
        stopper: Optional[EarlyStopper] = None,
    ):
        """
        Initializes reference(s) to the object(s) for early interruption of
        the optimization process upon certain condition(s).

        Args:
            timeout: instance of timeout checker or None.
            stopper: instance of early stopper or None.
        """
        self._service.set_status_trackers(timeout, stopper)

    @property
    def num_thetas(self) -> int:
        """
        Returns:
            number of parameters (angles) of rotation gates in this circuit.
        """
        return self._circuit.num_thetas

    @property
    def num_states(self) -> int:
        """
        Returns:
            number of states {|0>, X_1@|0>, X_2@|0>, ...} that make up
            the terms in objective function.
        """
        return self._num_states

    @property
    def target(self) -> Union[np.ndarray, QiskitMPS]:
        """
        Returns:
            a state being approximated: either Numpy array or MPS representation.
        """
        return self._target

    def set_target(self, target: Union[np.ndarray, QiskitMPS]) -> None:
        """
        Initializes internal reference to the target state.

        Args:
            target: a state to approximate in the optimization procedure;
                    either Numpy array or MPS representation.
        """
        if isinstance(target, np.ndarray):
            assert not self._use_mps
            assert chk.complex_1d(target)
            self._target = target
        else:
            assert self._use_mps and check_mps(target)
            assert len(target[0]) == self._circuit.num_qubits
            self._target = target

    @property
    def statistics(self) -> dict:
        """
        Returns:
            optimization statistics, if available, or empty dictionary.
        """
        return self._service.statistics

    def on_epoch_end(self):
        """
        Marks the end of optimization stage or epoch.
        """
        self._service.on_epoch_end()
