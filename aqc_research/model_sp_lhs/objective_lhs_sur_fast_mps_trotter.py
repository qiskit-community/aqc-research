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
Module defines the class for computation of surrogate objective function and
its gradient. Here we find the maximal projection onto one of the flip-states
or |0>, which makes up the second term in the objective function.
Most notably:
1) instead of full state vectors we use their MPS representations.
2) parametrized ansatz has a similar layout as the Trotter circuit,
    assuming layered structure; there is an older implementation in code
    archive that does not expect layered structure.
3) implementation is based on accelerated MPS objective.
"""

from typing import Tuple, Optional
import numpy as np
from aqc_research.parametric_circuit import (
    ParametricCircuit,
    TrotterAnsatz,
    layer_to_block_range,
    first_layer_included,
)
import aqc_research.utils as helper
import aqc_research.mps_operations as mpsop
from aqc_research.mps_dot_objective import fast_dot_gradient
import aqc_research.model_sp_lhs.objective_base as obj_base
from aqc_research.optimizer import GradientAmplifier

_logger = helper.create_logger(__file__)


class SpSurrogateObjectiveFastMpsTrotter(obj_base.SpLHSObjectiveBase):
    """
    Class for computation of surrogate objective function and its gradient.
    Here we find the maximal projection onto one of the flip-states or |0>,
    which makes up the second term in the objective function.
    Most notably:
    1) instead of full state vectors we use their MPS representations.
    2) parametrized ansatz has a similar layout as the Trotter circuit,
       assuming layered structure; there is an older implementation in code
       archive that does not expect layered structure.
    3) implementation is based on accelerated MPS objective.
    """

    _gamma = 0.1  # rate of exponential smoothing of the weighting factor

    def __init__(
        self,
        *,
        user_parameters: dict,
        circ: ParametricCircuit,
        layer_range: Optional[Tuple[int, int]] = None,
        alt_layers: bool = False,
        verbose: bool = False,
        grad_scaler: Optional[GradientAmplifier] = None,
    ):
        """
        Args:
            user_parameters: user supplied parameters.
            circ: instance of a parametric circuit.
            layer_range: couple of indices [from, to) that defines a range of layers
                         to compute gradients for; gradients of other layers will be
                         set to zero; default value (None) implies full range.
            alt_layers: enables alternating full optimization when
                        ``layer_range == None``; ignored and set to False in release
                        version; available only in developer version of this package.
            verbose: enables verbose status messages.
            grad_scaler: object that computes a scaling factor to amplify
                         gradient in the situation of barren plateau.
        """
        super().__init__(user_parameters, circ, use_mps=True, verbose=verbose)
        assert isinstance(circ, TrotterAnsatz)
        assert layer_range is None or isinstance(layer_range, tuple)
        assert isinstance(alt_layers, bool)
        assert grad_scaler is None or isinstance(grad_scaler, GradientAmplifier)

        if layer_range is None:
            layer_range = (0, circ.num_layers)  # full range
        assert len(layer_range) == 2

        # Disable alternating in release.
        if alt_layers:
            alt_layers = False
            _logger.warning(
                "in release version alternating optimization is disabled; "
                "'alt_layers' will be set to False"
            )

        self._trunc_thr = float(user_parameters["trunc_thr"])
        self._layer_range = layer_range
        self._state_prep_func = user_parameters.get("state_prep_func", None)
        self._fidelity = float(-1)
        self._grad_scaler = grad_scaler

        if self.num_states != self._circuit.num_qubits + 1:
            raise ValueError("only a single bit flip is currently supported")

        # Hilbert-Schmidt products <state|V.H|target> of the all states:
        self._hs = np.zeros(self._num_states, dtype=np.cfloat)

        # Index of the state of maximal projection (max. HS product):
        self._max_no = int(0)

    def objective(self, thetas: np.ndarray) -> float:
        """
        Computes the value of objective function given a vector of angular parameters.

        Args:
            thetas: current vector of angular parameters.

        Returns:
            a float value of the objective function.
        """
        # Store the latest encountered circuit parameters.
        self._store_latest_thetas(thetas)

        # This implementation handles a single bit flip only.
        assert mpsop.check_mps(self.target)
        assert self._circuit.num_qubits + 1 == self.num_states

        # Compute V.H @ target_state.
        self._vh_target = mpsop.v_dagger_mul_mps(
            circ=self._circuit,
            thetas=thetas,
            mps_vec=self.target,
            trunc_thr=self._trunc_thr,
        )

        # Compute the Hilbert-Schmidt products of all the states.
        for i in range(self.num_states):
            self._hs[i] = self._state_handler.state_dot_vector(i, self._vh_target)

        np.copyto(self._hs2, np.absolute(self._hs) ** 2)

        # Pick-up (flipped) state that gives the largest HS-product. We employ
        # a conservative approach here: the formerly leading state is changed,
        # if a new leader is really better (the idea of hysteresis).
        max_proj = self._hs2[self._max_no]
        for i in range(self.num_states):
            if 1.1 * max_proj < self._hs2[i]:
                max_proj = self._hs2[i]
                self._max_no = i

        # Surrogate objective function with the max-projection term.
        wgh = self._weight
        self._fobj = 1.0 - (1.0 - wgh) * self._hs2[0] - wgh * self._hs2[self._max_no]
        self._fidelity = self._hs2[0]

        self._service.on_end_objective()
        return self._fobj

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """
        Computes the gradient with respect to angular parameters given
        a vector of parameter values.

        Args:
            thetas: current vector of angular parameters.

        Returns:
            an array of gradient values.
        """
        self._service.on_begin_gradient(self._fobj, thetas, self._fidelity)
        self._calc_objective_before_gradient(thetas)

        circ = self._circuit
        layer_range = self._layer_range
        block_range = layer_to_block_range(circ, layer_range)
        optimize_front_layer = first_layer_included(circ, layer_range)

        # Contribution of the first term (with state |0>) to the gradient.
        grad_0 = fast_dot_gradient(
            circ=self._circuit,
            thetas=thetas,
            lvec=self._state_handler.init_state(0),
            vh_phi=self._vh_target,
            trunc_thr=self._trunc_thr,
            block_range=block_range,
            front_layer=optimize_front_layer,
        )

        # If max_no=0, we can save computations by calculating the gradient
        # for a single state |0>.
        if self._max_no == 0:
            grad_0 *= -2 * np.conj(self._hs[0])
            full_grad = grad_0.real.copy()  # copy makes the array contiguous
        else:
            grad_0 *= -2 * (1 - self._weight) * np.conj(self._hs[0])
            full_grad = grad_0.real.copy()  # copy makes the array contiguous

            # Contribution of the second term to the gradient.
            grad_max = fast_dot_gradient(
                circ=self._circuit,
                thetas=thetas,
                lvec=self._state_handler.init_state(self._max_no),
                vh_phi=self._vh_target,
                trunc_thr=self._trunc_thr,
                block_range=block_range,
                front_layer=optimize_front_layer,
            )

            grad_max *= -2 * self._weight * np.conj(self._hs[self._max_no])
            full_grad += grad_max.real

        # Amplify the gradient to prevent early termination on barren plateau.
        # Experimental.
        if self._grad_scaler:
            full_grad *= self._grad_scaler.estimate(self._fobj)

        # Gradually update the weight of the composite terms.
        gamma = self._gamma
        self._weight += gamma * (float(np.sqrt(abs(self._fobj))) - self._weight)

        self._service.on_end_gradient(
            self._fobj, self._fidelity, full_grad, self._hs2, self._weight
        )
        return full_grad

    @property
    def fidelity(self) -> float:
        """Returns the current fidelity measure."""
        return self._fidelity
