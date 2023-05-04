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
Modules defines the class for computation of surrogate objective function
and its gradient. Here we find the maximal projection onto one of the
flip-states or |0>, which makes up the second term in the objective function.
This is a classical approach where states are defined as full vectors.
"""

from typing import Tuple, Optional
import numpy as np
import aqc_research.utils as helper
import aqc_research.core_operations as cop
import aqc_research.model_sp_lhs.objective_base as obj_base
from aqc_research.parametric_circuit import ParametricCircuit
import aqc_research.checking as chk
from aqc_research.optimizer import GradientAmplifier

_logger = helper.create_logger(__file__)


class SpSurrogateObjectiveMax(obj_base.SpLHSObjectiveBase):
    """
    Class for computation of surrogate objective function and its gradient.
    Here we find the maximal projection onto one of the flip-states or |0>,
    which makes up the second term in the objective function.
    This is a classical approach where states are defined as full vectors.
    """

    _gamma = 0.1  # rate of exponential smoothing of the weighting factor

    def __init__(
        self,
        *,
        user_parameters: dict,
        circ: ParametricCircuit,
        block_range: Optional[Tuple[int, int]] = None,
        front_layer: bool = False,
        verbose: bool = False,
        grad_scaler: Optional[GradientAmplifier] = None,
    ):
        """
        Args:
            user_parameters: user supplied parameters.
            circ: instance of a parametric circuit.
            block_range: couple of indices [from, to) that defines a range of blocks
                         to compute gradients for; gradients of other blocks will be
                         set to zero; default value (None) implies full range.
            front_layer: enables optimization of the front layer of 1-qubit gates.
            verbose: enables verbose status messages.
            grad_scaler: object that computes a scaling factor to amplify
                         gradient in the situation of barren plateau.
        """
        super().__init__(user_parameters, circ, verbose=verbose)
        block_range = (0, circ.num_blocks) if block_range is None else block_range
        assert chk.is_tuple(block_range, len(block_range) == 2)
        assert 0 <= block_range[0] < block_range[1] <= circ.num_blocks
        assert chk.is_bool(front_layer)
        assert grad_scaler is None or isinstance(grad_scaler, GradientAmplifier)

        self._block_range = block_range
        self._front_layer = front_layer
        self._fidelity = float(-1)
        self._grad_scaler = grad_scaler

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

        # Compute V.H @ target_state.
        cop.v_dagger_mul_vec(
            circ=self._circuit,
            thetas=thetas,
            vec=self._target,
            out=self._vh_target,
            workspace=self._workspace[0:2],
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

        # Optimize the front layer in case of full gradient or user choice.
        optimize_front_layer = bool(
            self._front_layer or self._block_range == (0, self._circuit.num_blocks)
        )

        # Contribution of the first term (with state |0>) to the gradient.
        grad_0 = cop.grad_of_dot_product(
            circ=self._circuit,
            thetas=thetas,
            x_vec=self._state_handler.init_state(0),
            vh_y_vec=self._vh_target,
            workspace=self._workspace,
            block_range=self._block_range,
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
            grad_max = cop.grad_of_dot_product(
                circ=self._circuit,
                thetas=thetas,
                x_vec=self._state_handler.init_state(self._max_no),
                vh_y_vec=self._vh_target,
                workspace=self._workspace,
                block_range=self._block_range,
                front_layer=optimize_front_layer,
            )

            grad_max *= -2 * self._weight * np.conj(self._hs[self._max_no])
            full_grad += grad_max.real

        # Amplify the gradient to prevent early termination on barren plateau.
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
