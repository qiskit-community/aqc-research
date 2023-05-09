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
**Note**, L-BFGS can terminate early in case of small gradient, which can happen
on barren plateau. Combination of L-BFGS and gradient descent might work better.
"""

from time import perf_counter
from typing import Any, Optional, Union, Callable
import numpy as np
from qiskit.algorithms.optimizers import L_BFGS_B, ADAM, COBYLA, BOBYQA
from qiskit.algorithms.optimizers.optimizer import OptimizerResult
import aqc_research.checking as chk
from aqc_research.utils import create_logger
from aqc_research.parametric_circuit import ParametricCircuit

_logger = create_logger(__file__)


class StagnantOptimizationWarning(UserWarning):
    """Exception that indicates absence of optimization progress."""

    pass


class TimeoutStopper:
    """
    Class tracks the "timeout" status during optimization.
    It counts time from the moment of instantiation.
    """

    def __init__(self, *, time_limit: int):
        """
        Args:
            time_limit: time limit in seconds; there is no limit if non-positive.
        """
        assert chk.is_int(time_limit)
        self._end_time = int(-1)
        if time_limit > 0:
            self._end_time = int(round(perf_counter() + time_limit + 0.5))

    def check(self):
        """
        Checks if timeout is exceeded.

        Raises:
             TimeoutError if timeout had been exceeded.
        """
        if 0 < self._end_time < perf_counter():
            raise TimeoutError("Early termination: timeout")


class NotImproveStopper:
    """
    Class tracks the status of optimizer and interrupts it earlier,
    if objective function does not *decrease* after a number of iterations.
    """

    def __init__(self, *, num_iters: int, raise_ex: bool = True):
        """
        Args:
            num_iters: max. number of iterations without improvement.
            raise_ex: if ``True``, then raises ``StagnantOptimizationWarning``
                      exception in case of stagnant minimization process,
                      otherwise the function ``check()`` returns ``True`` when
                      no-improvement situation has been observed.
        """
        assert chk.is_int(num_iters, num_iters > 1)
        assert isinstance(raise_ex, bool)
        self._num_iters = int(num_iters)
        self._min_fobj = np.inf
        self._min_iteration = 0
        self._enabled = True
        self._raise_ex = raise_ex

    def reset(self):
        """Enforces a fresh start by discarding accumulated information."""
        self._min_fobj = np.inf
        self._min_iteration = 0
        self._enabled = True

    def disable(self):
        """Disables this object for detecting stagnant optimization process."""
        self._enabled = False

    def check(self, fobj: float, iter_no: int) -> bool:
        """
        Checks the stop condition for optimization.

        Args:
            fobj: objective function value.
            iter_no: index of optimization iteration.

        Returns:
            if ``raise_ex == False`` and there is no improvement of
            objective function value for a number of iterations, then returns
            ``True``, otherwise ``False`` or fires an exception.

        Raises:
             StagnantOptimizationWarning, if ``raise_ex == True`` and the stop
             condition has been met.
        """
        if not self._enabled:
            return False
        assert chk.is_float(fobj) and chk.is_int(iter_no, iter_no >= 0)
        if fobj < self._min_fobj:
            self._min_fobj = fobj
            self._min_iteration = iter_no
        elif iter_no - self._min_iteration > self._num_iters:
            if self._raise_ex:
                raise StagnantOptimizationWarning("Early termination, no improvement")
            return True
        return False


class SmallObjectiveStopper:
    """
    Class tracks the status of optimizer and interrupts it earlier,
    if the value of objective function fell below a threshold.
    """

    def __init__(self, *, fobj_thr: float):
        """
        Args:
            fobj_thr: threshold on objective function being minimized.
        """
        assert chk.is_float(fobj_thr)
        self._fobj_thr = fobj_thr

    def check(self, fobj: float):
        """
        Checks the stop condition for optimization.

        Args:
            fobj: objective function value.

        Raises:
             StopIteration if the stop conditions had been met.
        """
        assert chk.is_float(fobj)
        if fobj < self._fobj_thr:
            raise StopIteration(
                f"Early termination, objective fobj={fobj:0.5f} fell below the "
                f"threshold={self._fobj_thr:0.5f}"
            )


class TimeoutChecker:
    """
    Class tracks the "timeout" status during optimization and fires
    TimeoutError if time limit had been exceeded.
    This obsolete class is similar to ``TimeoutStopper`` but has extended
    functionality. It will be replaced by ``TimeoutStopper`` in the future.
    """

    def __init__(self, *, time_limit: Union[int, dict], start_immediately: bool = True):
        """
        Args:
            time_limit: time limit on optimization in seconds or dictionary
                        of parameters; there is no limit if non-positive.
            start_immediately: if True, the timeout interval is counted from
                               the time of creation of this instance.
        """
        if isinstance(time_limit, dict):
            time_limit = time_limit.get("timeout", -1)
        assert chk.is_int(time_limit) and chk.is_bool(start_immediately)
        self._end_time = -1
        self._time_limit = time_limit
        self._results = dict({})
        if start_immediately:
            self.start()

    def start(self):
        """
        Commences the timeout interval using the current time.
        """
        now = int(round(perf_counter() + 0.5))
        self._end_time = -1 if self._time_limit <= 0 else now + self._time_limit

    def check(
        self,
        fobj: float,
        thetas: np.ndarray,
        on_stop: Optional[Callable[[float, np.ndarray], dict]] = None,
    ):
        """
        Checks if timeout is exceeded. Stores the best solution achieved right
        before the timeout, if ``on_stop()`` function is specified.

        Args:
            fobj: current value of objective function.
            thetas: current angular parameters.
            on_stop: callable that will be invoked before TimeoutError is fired;
                     it takes the current objective function value and angular
                     parameters as arguments, and returns a dictionary of
                     optimization results.

        Raises:
             TimeoutError if timeout had been exceeded.
        """
        if 0 < self._end_time < perf_counter():
            if on_stop is not None:
                assert callable(on_stop)
                results = on_stop(fobj, thetas)
                assert chk.is_dict(results)
                self._results = results
            raise TimeoutError("early termination: timeout")

    @property
    def optim_results(self) -> dict:
        """
        Returns a dictionary with (the best) results achieved by optimization
        process right before it had been interrupted by timeout.
        """
        return self._results


class EarlyStopper:
    """
    Class tracks the status of optimizer and interrupts it earlier, if either
    the objective function does not go down after a number of iterations
    or its value fell below a threshold or fidelity exceeded a threshold.
    This obsolete class combines functionality of ``NoImproveStopper`` and
    ``SmallObjectiveStopper``. It will be replaced by those classes in the future.
    """

    def __init__(
        self,
        fobj_thr: Optional[float] = None,
        fidelity_thr: Optional[float] = None,
        num_iters: Optional[int] = None,
    ):
        """
        Args:
            fobj_thr: threshold on objective function; should be a small value.
            fidelity_thr: threshold on fidelity value;
                          should be less than but close to 1.
            num_iters: max. number of iterations without improvement.
        """
        super().__init__()
        assert fobj_thr is None or chk.is_float(fobj_thr)
        assert fidelity_thr is None or chk.is_float(fidelity_thr)
        assert fidelity_thr is None or 0 < fidelity_thr <= 1
        assert num_iters is None or chk.is_int(num_iters)
        # assert (fobj_thr is not None) or (num_iters is not None)

        self._fobj_thr = fobj_thr
        self._fidelity_thr = fidelity_thr
        self._early_stop_iters = num_iters if num_iters else -1
        self._min_fobj = np.inf
        self._min_thetas = np.empty(0)
        self._min_iteration = 0
        self._results = dict({})

    def check(
        self,
        fobj: Union[float, None],
        fidelity: Union[float, None],
        thetas: np.ndarray,
        iter_no: int,
        on_stop: Callable[[float, np.ndarray], dict],
    ):
        """
        Checks a stop condition for optimization.

        Args:
            fobj: objective function value or None (ignored).
            fidelity: fidelity value or None (ignored).
            thetas: current angular parameters.
            iter_no: index of optimization iteration.
            on_stop: callable that will be invoked before StopIteration is fired;
                     it takes the current objective function value and angular
                     parameters as arguments, and returns a dictionary of
                     optimization results.

        Raises:
             StopIteration if a stop conditions had been met.
        """
        assert chk.none_or_type(fobj, float) and chk.none_or_type(fidelity, float)
        assert chk.float_1d(thetas) and chk.is_int(iter_no) and callable(on_stop)

        if self._min_thetas.size == 0:
            self._min_thetas = thetas.copy()

        # Check the objective function became sufficiently small.
        if (fobj is not None) and (self._fobj_thr is not None):
            if fobj < self._fobj_thr:
                self._set_optim_results(on_stop(fobj, thetas))
                raise StopIteration(
                    f"early termination, objective fobj={fobj:0.5f} fell below the "
                    f"threshold={self._fobj_thr:0.5f}"
                )

        # Check if there is no improvement after a number of iterations.
        if (fobj is not None) and self._early_stop_iters > 0:
            if fobj < self._min_fobj:
                self._min_fobj = fobj
                np.copyto(self._min_thetas, thetas)
                self._min_iteration = iter_no
            elif iter_no - self._min_iteration > self._early_stop_iters:
                self._set_optim_results(on_stop(self._min_fobj, self._min_thetas))
                raise StopIteration("Early termination, no improvement")

        # Check the fidelity became sufficiently large.
        if (fidelity is not None) and (self._fidelity_thr is not None):
            if fidelity >= self._fidelity_thr:
                self._set_optim_results(on_stop(fobj, thetas))
                raise StopIteration(
                    f"early termination, fidelity={fidelity:0.3f} exceeded "
                    f"the threshold={self._fidelity_thr:0.3f}"
                )

    @property
    def optim_results(self) -> dict:
        """
        Returns a dictionary with (the best) results achieved by optimization
        process before it had been interrupted.
        """
        return self._results

    def _set_optim_results(self, results: dict):
        """
        Stores a dictionary with the best so far optimization results.
        """
        assert chk.is_dict(results)
        self._results = results


class GradientAmplifier:
    """
    Class dynamically estimates the gradient amplification multiplier in
    situation where objective function struggles in barren plateau and its
    gradient becomes very small (so-called "vanishing gradient" problem).
    We compute the gradient scaling coefficient using a logarithmic function,
    which provides a moderate amplification without excessively harmful
    effect on optimization algorithm. E.X.P.E.R.I.M.E.N.T.A.L

    **Note**, the gradient amplifier can screw up an optimization process.
    Its usefulness is rather arguable. One should use it cautiously.

    **Note**, this class was designed for an objective function varying between
    0 and 1 (or in a comparable range). Correction might be needed if the range
    of objective function significantly differs from 1.
    """

    def __init__(self, history: int = 5, strong: bool = False, verbose: bool = False):
        """
        Args:
            history: number of recent samples of objective function, which
                     are used to estimate the deviation of objective function.
            strong: enables natural logarithm in amplification formula with
                    stronger gradient boost; otherwise log10 will be used
                    with softer magnification.
            verbose: enables verbosity.
        """
        assert chk.is_int(history, history >= 3)
        assert isinstance(strong, bool) and isinstance(verbose, bool)
        self._history = np.zeros(history)
        self._counter = np.int64(0)
        self._logarithm = np.log if strong else np.log10
        self._scale = 1.0
        self._verbose = verbose
        if verbose:
            _logger.warning("enabled gradient amplification in case of barren plateau")

    def estimate(self, fobj: float) -> float:
        """
        Updates the history of recent objective function values and deduces
        gradient amplification coefficient from sample deviation.

        Args:
            fobj: current value of objective function.

        Returns:
            gradient amplification coefficient.
        """
        # Replace the oldest sample in the recent history.
        self._history[self._counter % self._history.size] = fobj
        self._counter += 1
        if self._counter < self._history.size:
            return 1.0  # history is incomplete yet

        dev = float(np.ptp(self._history))  # "peak to peak" diff. in history
        new_scale = max(-float(self._logarithm(max(dev, 1e-8))), 1.0)
        self._scale += 0.3 * (new_scale - self._scale)  # some exp. smoothing
        if self._verbose and self._scale > 1.5:
            _logger.info("gradient scale: %0.4f", self._scale)
        return self._scale


class AQCOptimResult:
    """
    Class keeps and updates optimization result as required by AQC framework.
    """

    def __init__(self, circ: ParametricCircuit, thetas_0: np.ndarray):
        """
        Args:
            circ: parametrized ansatz.
            thetas_0: initial vector of angular parameters.
        """
        self._result = {
            "cost": float(1e30),
            "num_iters": int(0),
            "num_fun_ev": int(0),
            "num_grad_ev": int(0),
            "ini_thetas": thetas_0.copy(),
            "thetas": thetas_0.copy(),
            "blocks": circ.blocks.copy(),
            "entangler": circ.entangler,
            "stats": dict({}),
        }

    def update_from_optimizer(self, res: OptimizerResult, blocks: np.ndarray):
        """
        Updates the result from Qiskit optimizer output.

        **Note**: circuit structure can in principle change over time,
        therefore we always save "blocks".

        **Note**: iteration counters are incremented, *not* overwritten,
        because optimization process can include several epochs or stages.

        Args:
            res: output of a Qiskit optimizer.
            blocks: structure of parametric circuit (positions of unit blocks).
        """
        assert isinstance(res, OptimizerResult)
        assert chk.int_2d(blocks, blocks.shape == (2, blocks.size // 2))
        self._result["cost"] = res.fun
        self._result["num_iters"] += res.nit if res.nit is not None else 0
        self._result["num_fun_ev"] += res.nfev if res.nfev is not None else 0
        self._result["num_grad_ev"] += res.njev if res.njev is not None else 0
        self._result["thetas"] = res.x.copy()
        self._result["blocks"] = blocks.copy()

    def update_from_dict(self, res: dict):
        """
        Updates the result from dictionary. Useful when optimization has been
        interrupted by some stopping condition, and we want to collect the
        best solution attained so far. Also, this function can be used for
        updating an individual fields such like statistics, for example.

        **Note**: it is assumed that user keeps track of iteration counters,
        angular parameters and circuit structure, which can be overwritten
        in this function from the external dictionary.

        Args:
            res: external dictionary of optimization result.
        """
        assert isinstance(res, dict)
        self._result.update(res)

    @property
    def thetas(self) -> np.ndarray:
        """
        Returns the vector of angular parameters.
        """
        return self._result["thetas"]

    @property
    def as_dict(self) -> dict:
        """
        Returns a dictionary of result data.
        """
        return self._result


class AqcOptimizer:
    """
    Class organizes the AQC/ASP optimization process by handling early
    termination upon either a timeout or other stop condition. It also collects
    statistics from the objective function at the end, if the statistics
    is provided. An instance of this class is very lightweight and can be
    passed around with negligible overhead.
    """

    _optimizers = [
        "adam",
        "lbfgs",
        "cobyla",
        "bobyqa",
    ]

    def __init__(
        self,
        *,
        optimizer_name: Optional[str] = "lbfgs",
        maxiter: Optional[int] = 1000,
        learn_rate: Optional[float] = 0.1,
        lbfgs_maxcor: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Args:
            optimizer_name: name of optimizer, one of
                            {"adam", "lbfgs", "cobyla", "bobyqa"}.
            maxiter: maximum number of optimization iterations.
            learn_rate: learning rate for Adam and AdaGrad gradient descent.
            lbfgs_maxcor: history size for L-BFGS method or None (defaults to 10).
            verbose: enables verbose status messages.
        """
        assert chk.is_str(optimizer_name, optimizer_name in self._optimizers)
        assert chk.is_int(maxiter, maxiter > 0)
        assert chk.is_float(learn_rate, 0 < learn_rate < 1)
        assert lbfgs_maxcor is None or chk.is_int(lbfgs_maxcor, lbfgs_maxcor > 0)
        assert chk.is_bool(verbose)

        self._optimizer_name = optimizer_name
        self._maxiter = maxiter
        self._learn_rate = learn_rate
        self._lbfgs_maxcor = lbfgs_maxcor  # scipy, minimize(method='L-BFGS-B')
        self._verbose = verbose

    def optimize(
        self,
        objv: Any,
        circ: ParametricCircuit,
        thetas_0: np.ndarray,
        *,
        stopper: Optional[EarlyStopper] = None,
        timeout: Optional[TimeoutChecker] = None,
    ) -> dict:
        """
        Runs optimization according to user choice.

        Args:
            objv: instance of objective function class that implements the
                  following methods: ``objective()``, ``gradient()``,
                  ``set_status_trackers()``, and optionally the property
                  functions ``statistics``, ``fidelity``.
            circ: instance of parametric circuit.
            thetas_0: initial angular parameters of parametric circuit.
            stopper: object for early termination according to some criterion.
            timeout: object for early termination upon timeout.

        Returns:
            the dictionary of the following results, see AQCOptimResult class:
            (1) "cost" - final value of objective function.
            (2) "num_iters" - number of iterations made.
            (3) "num_fun_ev" - number of objective function evaluations.
            (4) "num_grad_ev" - number of gradient of obj. function evaluations.
            (5) "ini_thetas" - initial angular parameters.
            (6) "thetas" - final angular parameters (solution).
            (7) "blocks" - final placements of 2-qubit unit blocks.
            (8) "stats" - instance of an object that comprises statistics or None.
            (9) "is_timeout" - True in case of timeout event, otherwise False.
            (10) "fidelity" value, if the objective implements ``fidelity``
                 property function; otherwise "fidelity" is not presented.
        """
        assert hasattr(objv, "objective")
        assert hasattr(objv, "gradient")
        assert hasattr(objv, "set_status_trackers")
        assert isinstance(circ, ParametricCircuit)
        assert chk.float_1d(thetas_0)
        assert chk.none_or_type(stopper, EarlyStopper)
        assert chk.none_or_type(timeout, TimeoutChecker)

        result = AQCOptimResult(circ, thetas_0)
        opname = self._optimizer_name
        is_timeout = False

        try:
            objv.set_status_trackers(timeout=timeout, stopper=stopper)
            self._log_message(f"running {opname.upper()} optimizer ...")
            if opname == "adam":
                self._log_message(f"learning rate: {self._learn_rate:0.6f}")
                optimizer = ADAM(maxiter=self._maxiter, lr=self._learn_rate)
                res = optimizer.minimize(fun=objv.objective, x0=thetas_0, jac=objv.gradient)
            elif opname == "lbfgs":
                lbfgs_options = dict({})
                if self._lbfgs_maxcor:
                    lbfgs_options["maxcor"] = self._lbfgs_maxcor

                optimizer = L_BFGS_B(
                    maxfun=5 * self._maxiter,
                    maxiter=self._maxiter,
                    options=lbfgs_options,
                )
                res = optimizer.minimize(fun=objv.objective, x0=thetas_0, jac=objv.gradient)
            elif opname == "cobyla":
                optimizer = COBYLA(maxiter=self._maxiter, tol=0.001)
                res = optimizer.minimize(fun=objv.objective, x0=thetas_0)
            elif opname == "bobyqa":
                optimizer = BOBYQA(maxiter=self._maxiter)
                bounds = [(-2 * np.pi, 2 * np.pi)] * thetas_0.size
                res = optimizer.minimize(fun=objv.objective, x0=thetas_0, bounds=bounds)
            else:
                raise ValueError(
                    f"unsupported optimizer: {opname}, expects one of: {self._optimizers}"
                )
            result.update_from_optimizer(res, circ.blocks)

        except StopIteration as ex:
            self._log_message(str(ex))
            if hasattr(objv, "optim_results"):
                result.update_from_dict(objv.optim_results)
            else:
                result.update_from_dict(stopper.optim_results)
        except TimeoutError as ex:
            is_timeout = True
            self._log_message(str(ex))
            if hasattr(objv, "optim_results"):
                result.update_from_dict(objv.optim_results)
            else:
                result.update_from_dict(timeout.optim_results)
        finally:
            result.update_from_dict({"is_timeout": is_timeout})
            if hasattr(objv, "fidelity"):
                result.update_from_dict({"fidelity": objv.fidelity})

        # Collect the final optimization statistics, if available.
        if hasattr(objv, "statistics"):
            stats = {"stats": objv.statistics}
            stats["stats"]["is_timeout"] = is_timeout
            result.update_from_dict(stats)

        return result.as_dict

    def _log_message(self, msg: str):
        """Prints out a message, if verbosity is enabled."""
        if self._verbose:
            _logger.info(msg)
