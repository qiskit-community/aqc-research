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
Core classes and routines needed to implement the full and sketched approaches
to approximate quantum compiling.
"""

from time import perf_counter
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional
import logging
import numpy as np
from aqc_research.parametric_circuit import ParametricCircuit
import aqc_research.checking as chk
from aqc_research.core_op_matrix import v_dagger_mul_mat, grad_of_matrix_dot_product
from aqc_research.optimizer import (
    GradientAmplifier,
    TimeoutStopper,
    NotImproveStopper,
    SmallObjectiveStopper,
)


class SketchingVectorsBase(ABC):
    """
    Abstract base class for any implementation of sketching vectors generator.
    Here we generate two matrices of sketching vectors stacked in columns.
    Namely, the matrix ``X`` of size ``(2^n)x(number of sketching vectors)``
    and corresponding matrix ``Y = U @ X``, where ``U`` is the target matrix.
    Both matrices participate in the product ``<X|V.H U|X> = <X|V.H|Y>`` -
    the main ingredient of different objective functions.
    """

    def __init__(self, num_skvecs: int, target_mat: Union[np.ndarray, np.memmap]):
        """
        Args:
            num_skvecs: number of sketching vectors; must be a power of 2 number.
            target_mat: target unitary (or SU) matrix.
        """
        assert chk.is_int(num_skvecs)
        assert chk.complex_2d_square(target_mat)
        num_skvecs = min(max(num_skvecs, 1), target_mat.shape[0])
        if not (num_skvecs > 0 and ((num_skvecs - 1) & num_skvecs) == 0):
            raise ValueError("'num_skvecs' must be a power of 2 number")

        self._num_skvecs = num_skvecs
        self._target_mat = target_mat

    @property
    def num_skvecs(self) -> int:
        """Returns the number of sketching vectors."""
        return self._num_skvecs

    @property
    def target_matrix(self) -> Union[np.ndarray, np.memmap]:
        """Returns the target matrix."""
        return self._target_mat

    @abstractmethod
    def generate(
        self,
        circ: Optional[ParametricCircuit] = None,
        thetas: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates sketching vectors. Should be implemented in a derived class.

        **Note**, in case derived class returns sketching vectors stored
        internally as class variables, do *not* assume they will remain unchanged,
        in fact, they will be modified by the gradient computation routine.

        Args:
            circ: optional parametrized ansatz.
            thetas: optional angular parameters of parametrized ansatz.

        Returns:
            (1) 2D array of sketching vectors ``X`` stacked in columns.
            (2) 2D array of sketching vectors multiplied by the target
                matrix ``Y = U @ X``, also stacked in columns.
        """
        raise NotImplementedError("abstract method")


class SketchingObjectiveEx:
    """
    Class for computation of objective function and its gradient.
    ``fobj = 1 - (1/num_skvecs) * Re(Tr(<V@Q,U@Q>))``,
    where ``Q`` is a matrix of sketching vectors, ``U`` is the target SU
    matrix, and ``V = V(thetas)`` is the matrix of parametrized ansatz.
    """

    def __init__(
        self,
        circ: ParametricCircuit,
        skvecs: SketchingVectorsBase,
        *,
        enable_stats: bool = False,
        grad_scaler: Optional[GradientAmplifier] = None,
        stop_timeout: Optional[TimeoutStopper] = None,
        stop_stagnant: Optional[NotImproveStopper] = None,
        stop_small_fobj: Optional[SmallObjectiveStopper] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            circ: instance of parametrized ansatz.
            skvecs: generator of sketching vectors; this object also provides
                    a reference to the target matrix.
            enable_stats: enables collecting the convergence statistics.
            grad_scaler: optional object for scaling the gradient.
            stop_timeout: optional for early termination upon timeout.
            stop_stagnant: optional checker for early termination of stagnant
                           optimization (in case of prolonged barren plateau).
            stop_small_fobj: optional checker for early termination upon
                             objective fell below a threshold.
            logger: optional logging object.
        """
        assert isinstance(circ, ParametricCircuit)
        assert isinstance(skvecs, SketchingVectorsBase)
        assert isinstance(enable_stats, bool)
        assert chk.none_or_type(grad_scaler, GradientAmplifier)
        assert chk.none_or_type(stop_timeout, TimeoutStopper)
        assert chk.none_or_type(stop_stagnant, NotImproveStopper)
        assert chk.none_or_type(stop_small_fobj, SmallObjectiveStopper)
        assert chk.none_or_type(logger, logging.Logger)

        dim, num_skvecs = circ.dimension, skvecs.num_skvecs
        self._circ = circ
        self._target = skvecs.target_matrix
        self._skvecs = skvecs
        self._enable_stats = enable_stats
        self._workspace = np.zeros((dim, num_skvecs), dtype=np.cfloat)
        self._grad_scaler = grad_scaler
        self._stop_timeout = stop_timeout
        self._stop_stagnant = stop_stagnant
        self._stop_small_fobj = stop_small_fobj
        self._logger = logger

        # The best so far value of objective function and angular parameters:
        self._fobj_best = float(np.inf)
        self._thetas_best = np.zeros(circ.num_thetas)

        # Statistics:
        self._nit = int(0)  # current number of iterations
        self._fobj_profile = list([])  # convergence profile of "fobj"

        # These variables are needed to interface with Qiskit optimizers:
        self._fobj_latest = float(1e30)  # the last computed objective's value
        self._grad_latest = np.empty(0)  # the last computed objective's gradient
        self._thetas_latest = np.empty(0)  # the last used angular parameters
        self._print_grad_warning = bool(logger is not None)

        # Variables for printing the progress:
        self._elapsed_time = perf_counter()
        self._period = int(round(10 + 60.0 / (1 + 2.0 ** (6 - circ.num_qubits))))

    def objective_and_gradient(self, thetas: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Computes the value of objective function and its gradient given
        the current angular parameters ``thetas``.

        Args:
            thetas: current angular parameters of ansatz.

        Returns:
            value of objective function and its gradient.
        """
        # Print progress.
        now = perf_counter()
        if self._elapsed_time + self._period < now:
            print(".", end="", flush=True)
            self._elapsed_time = now

        circ = self._circ
        num_skvecs = self._skvecs.num_skvecs
        worksp = self._workspace

        # Recompute objective and gradient given the sketching vectors.
        # vh_y = V.H|y>, y = U @ x, fobj = <x|V.H|y>, grad = <x|d(V.H)/d(theta)|y>
        x, y = self._skvecs.generate(circ, thetas)
        vh_y = v_dagger_mul_mat(circ, thetas, y, worksp)
        fobj = 1 - np.real(np.vdot(x, vh_y)) / num_skvecs
        grad = grad_of_matrix_dot_product(circ, thetas, x, vh_y, worksp)
        grad = -np.real(grad) / num_skvecs

        # Amplify a vanishing gradient. Experimental.
        if self._grad_scaler:
            grad *= self._grad_scaler.estimate(fobj)

        # Memorize the best so far objective function and corresponding thetas.
        if fobj < self._fobj_best:
            self._fobj_best = fobj
            np.copyto(self._thetas_best, thetas)

        # Collect the statistics.
        self._nit += 1
        if self._enable_stats:
            self._fobj_profile.append(float(fobj))
        if self._logger is not None:
            # Printing not logging because we just want to see the progress.
            gnorm = np.linalg.norm(grad)
            print(f"\riter: {self._nit:4d}, fobj: {fobj:0.4f}, |grad|: {gnorm:0.5f}")

        # Check the stop conditions.
        if self._stop_timeout:
            self._stop_timeout.check()
        if self._stop_stagnant:
            self._stop_stagnant.check(fobj=fobj, iter_no=self._nit)
        if self._stop_small_fobj:
            self._stop_small_fobj.check(fobj=fobj)

        return fobj, grad

    def objective(self, thetas: np.ndarray) -> float:
        """
        Computes the value of objective function given the current angular
        parameters ``thetas``. Note, Qiskit optimizers expect the separate
        functions for computation of the objective and its gradient. We need
        this function to conform to Qiskit interface.

        Args:
            thetas: current angular parameters of ansatz.

        Returns:
            value of objective function.
        """
        if self._thetas_latest.size == 0:
            self._thetas_latest = thetas.copy()
        else:
            np.copyto(self._thetas_latest, thetas)

        self._fobj_latest, self._grad_latest = self.objective_and_gradient(thetas)
        return self._fobj_latest

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of objective function given the current angular
        parameters ``thetas``. Note, Qiskit optimizers expect the separate
        functions for computation of the objective and its gradient. We need
        this function to conform to Qiskit interface.

        Args:
            thetas: current angular parameters of ansatz.

        Returns:
            gradient of objective function.
        """
        tol = float(10.0 * np.finfo(thetas.dtype).eps)
        last = self._thetas_latest
        if last.size == 0 or not np.allclose(thetas, last, atol=tol, rtol=tol):
            self.objective(thetas)

        return self._grad_latest

    @property
    def statistics(self) -> dict:
        """
        Returns a dictionary with optimization statistics, if available.
        """
        return {
            "convergence_profile": np.asarray(self._fobj_profile, dtype=np.float32),
            "nit": self._nit,
        }

    @property
    def num_iterations(self) -> int:
        """Returns the number of iterations made so far."""
        return int(self._nit)

    @property
    def optim_results(self) -> dict:
        """
        Returns a dictionary with the best optimization result achieved so far.
        """
        return {
            "cost": float(self._fobj_best),
            "num_fun_ev": int(self._nit),
            "num_grad_ev": int(self._nit),
            "num_iters": int(self._nit),
            "thetas": self._thetas_best,
            "entangler": self._circ.entangler,
            "blocks": self._circ.blocks.copy(),
        }

    def set_status_trackers(self, timeout, stopper):
        """Compatibility function for AqcOptimizer class."""
        pass


class FullRangeSketchingVectors(SketchingVectorsBase):
    """
    Suits for the full AQC optimization ``<X|V.H U|X>``, where ``X`` is a square
    matrix of size ``2^n x 2^n`` (same as ``V`` and ``U``), and the sketching
    vectors span the entire column space of the target matrix ``U``.
    """

    def __init__(self, target_mat: Union[np.ndarray, np.memmap]):
        """
        Args:
            target_mat: target unitary matrix.
        """
        super().__init__(target_mat.shape[0], target_mat)
        dim = self.target_matrix.shape[0]
        self._x_vecs = np.zeros((dim, self.num_skvecs), dtype=np.cfloat)
        self._y_vecs = np.zeros_like(self._x_vecs)

    def generate(
        self,
        _: Optional[ParametricCircuit] = None,
        __: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates sketching vectors, see the base class description."""
        self._x_vecs.fill(0)
        np.fill_diagonal(self._x_vecs, 1)  # x = I
        np.copyto(self._y_vecs, self.target_matrix)  # y = U @ x = U @ I = U
        return self._x_vecs, self._y_vecs


class RandomSketchingVectors(SketchingVectorsBase):
    """
    Draws the new random sketching vectors upon every request.
    """

    def __init__(self, num_skvecs: int, target_mat: Union[np.ndarray, np.memmap]):
        """
        Args:
            num_skvecs: number of sketching vectors; must be a power of 2 value.
            target_mat: target unitary (or SU) matrix.
        """
        super().__init__(num_skvecs, target_mat)
        dim = self.target_matrix.shape[0]
        assert dim % self.num_skvecs == 0
        self._storage = np.zeros((dim, self.num_skvecs), dtype=np.cfloat)

    def generate(
        self,
        _: Optional[ParametricCircuit] = None,
        __: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates sketching vectors, see the base class description."""
        dim, num_skvecs = self.target_matrix.shape[0], self.num_skvecs
        x_vecs, _ = np.linalg.qr(
            np.random.rand(dim, num_skvecs) + 1j * np.random.rand(dim, num_skvecs)
        )
        y_vecs = np.dot(self.target_matrix, x_vecs, out=self._storage)
        return x_vecs, y_vecs


class AlternatingSketchingVectors(SketchingVectorsBase):
    """
    Draws a random subset of target matrix's columns to be used as
    the sketching vectors upon every request.
    """

    def __init__(self, num_skvecs: int, target_mat: Union[np.ndarray, np.memmap]):
        """
        Args:
            num_skvecs: number of sketching vectors; must be a power of 2 value.
            target_mat: target unitary (or SU) matrix.
        """
        super().__init__(num_skvecs, target_mat)
        dim = target_mat.shape[0]
        assert dim % self.num_skvecs == 0
        self._offset = int(0)
        self._indices = np.random.permutation(dim)
        self._x_vecs = np.zeros((dim, self.num_skvecs), dtype=np.cfloat)
        self._y_vecs = np.zeros((dim, self.num_skvecs), dtype=np.cfloat)

    def generate(
        self,
        _: Optional[ParametricCircuit] = None,
        __: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates sketching vectors, see the base class description."""
        target = self.target_matrix
        dim = target.shape[0]

        # Generate a new permutation upon cycle completion.
        if self._offset >= dim:
            self._offset = 0
            self._indices = np.random.permutation(dim)

        # Get the next subsets of sketching vectors X and Y = U @ X.
        idx = self._indices[self._offset : self._offset + self.num_skvecs]
        self._x_vecs.fill(0)
        for i in range(idx.size):
            self._x_vecs[idx[i], i] = 1  # make a unit vector (0,..,1,..,0).T
            np.copyto(self._y_vecs[:, i], target[:, idx[i]])  # y_i = U @ x_i

        self._offset += self.num_skvecs
        return self._x_vecs, self._y_vecs


class EigenSketchingVectors(SketchingVectorsBase):
    """
    If we consider objective function in the form:
    ``fobj = const * |Q @ Q.H @ (V.H - U.H)|^2``, then, by formal expansion,
    it straightforward to show that (using a proper constant ``const``):
    ``fobj = 1 - (1/num_skvecs) * Re(Tr(<V@Q,U@Q>))``, which is exactly
    the function we minimize in the above objective class.
    In this case, it might be reasonable to choose sketching vectors as
    the largest singular ones of the difference (V.H - U.H), i.e.
    we focus our attention on the subspace of the biggest discrepancies.
    The paper by Halko et al., 2010, https://arxiv.org/pdf/0909.4061.pdf
    gives efficient method for the partial SVD.
    """

    def __init__(self, num_skvecs: int, target_mat: Union[np.ndarray, np.memmap]):
        """
        Args:
            num_skvecs: number of sketching vectors; must be a power of 2 value.
            target_mat: target unitary (or SU) matrix.
        """
        super().__init__(num_skvecs, target_mat)
        dim = target_mat.shape[0]
        self._workspace = np.zeros((3, dim, num_skvecs), dtype=np.cfloat)

    def generate(
        self,
        circ: Optional[ParametricCircuit] = None,
        thetas: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates sketching vectors, see the base class description."""
        assert isinstance(circ, ParametricCircuit)
        assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
        assert circ.dimension == self.target_matrix.shape[0]

        dim, num_skvecs = circ.dimension, self.num_skvecs
        target = self.target_matrix
        worksp = self._workspace
        omega = worksp[0]

        # Random, normally distributed complex vectors stacked in columns.
        np.copyto(omega, np.random.randn(dim, num_skvecs))
        omega *= 1j
        omega += np.random.randn(dim, num_skvecs)

        # uh_omega := U.H @ omega = (omega.H @ U).H  (goes in worksp[1]).
        _out1 = worksp[1].reshape(num_skvecs, dim)  # "transposed" worksp[1]
        _out2 = worksp[2].reshape(num_skvecs, dim)  # "transposed" worksp[2]
        uh_omega = np.dot(np.conj(omega.T, out=_out1), target, out=_out2)
        uh_omega = np.conj(uh_omega.T, out=worksp[1])
        assert uh_omega.shape == (dim, num_skvecs)

        # vuh_omega := (V.H - U.H) @ omega  (goes in worksp[0]).
        vuh_omega = v_dagger_mul_mat(circ, thetas, omega, worksp[2])
        vuh_omega -= uh_omega
        assert vuh_omega.shape == (dim, num_skvecs)

        # Forge X and Y = U @ X vectors  (y is formed in worksp[0]).
        x_vecs, _ = np.linalg.qr(vuh_omega)
        assert x_vecs.shape == (dim, num_skvecs)
        y_vecs = np.dot(target, x_vecs, out=worksp[0])
        return x_vecs, y_vecs


def skvecs_generator(
    skvecs_type: str, num_skvecs: int, target_mat: Union[np.ndarray, np.memmap]
) -> SketchingVectorsBase:
    """
    Instantiates chosen generator of sketching vectors.

    Args:
        skvecs_type: one of ['full', 'rand', 'alt', 'eigen'].
        num_skvecs: number of sketching vectors; must be a power of 2 value.
        target_mat: target unitary (or SU) matrix.

    Returns:
        instance of sketching vectors generator.
    """
    assert isinstance(skvecs_type, str)
    if skvecs_type == "full" or num_skvecs == target_mat.shape[0]:
        return FullRangeSketchingVectors(target_mat)
    elif skvecs_type == "rand":
        return RandomSketchingVectors(num_skvecs, target_mat)
    elif skvecs_type == "alt":
        return AlternatingSketchingVectors(num_skvecs, target_mat)
    elif skvecs_type == "eigen":
        return EigenSketchingVectors(num_skvecs, target_mat)
    else:
        raise ValueError(
            f"unknown type of sketching vectors generator, expects one of: "
            f"['full', 'rand', 'alt', 'eigen'], got {skvecs_type}"
        )
