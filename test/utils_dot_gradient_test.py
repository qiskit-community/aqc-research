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
Building blocks for any gradient unit test of a dot-product objective:
``fobj = <x|V.H|y>``, where ``x`` and ``y`` are some state vectors and
``V = V(thetas)`` is a matrix of parametric circuit (V.H means adjoint).
"""

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Callable, List, Dict, Tuple, Optional
import numpy as np
from aqc_research.parametric_circuit import ParametricCircuit, TrotterAnsatz
import test.utils_for_testing as tut
import aqc_research.utils as helper
from aqc_research.circuit_structures import create_ansatz_structure, make_trotter_like_circuit
import aqc_research.checking as chk


class BaseGradTestObjective(ABC):
    """Base class of any objective function designed for gradient testing."""

    @abstractmethod
    def is_trotterized_ansatz(self) -> bool:
        """Returns non-zero if Trotterized ansatz is being used."""
        raise NotImplementedError()

    def trotter_order(self) -> bool:
        """Returns non-zero if 2nd order Trotterized ansatz was selected."""
        raise NotImplementedError()

    @abstractmethod
    def initialize(self, circ: ParametricCircuit, thetas: np.ndarray, close_states: bool):
        """Initializes this object."""
        raise NotImplementedError()

    @abstractmethod
    def objective_from_matrix(self, thetas: np.ndarray) -> np.cfloat:
        """Computes objective from the circuit matrix built by Qiskit."""
        raise NotImplementedError()

    @abstractmethod
    def objective(self, thetas: np.ndarray) -> np.cfloat:
        """Computes objective from parametric circuit directly."""
        raise NotImplementedError()

    @abstractmethod
    def gradient(
        self,
        thetas: np.ndarray,
        block_range: Optional[Tuple[int, int]] = None,
        front_layer: Optional[bool] = True,
    ) -> np.ndarray:
        """Computes gradient from parametric circuit directly."""
        raise NotImplementedError()


def _numerical_gradient(
    objv_func: Callable[[np.ndarray], float],
    thetas: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Computes numerical gradient of the objective function.
    """
    th_tau = thetas.copy()
    grad = np.zeros(thetas.size, dtype=np.cfloat)
    for i in range(thetas.size):
        th_tau[i] = thetas[i] - tau
        f_m = objv_func(th_tau)
        th_tau[i] = thetas[i] + tau
        f_p = objv_func(th_tau)
        grad[i] += (f_p - f_m) / (2.0 * tau)
        th_tau[i] = thetas[i]
    return grad


def partial_vs_full_gradient_test(conf: dict, objv: BaseGradTestObjective) -> dict:
    """
    Compares full and partial gradients by randomly inserting (random) extra
    unit blocks in (randomly generated) parametric circuit and computing the
    partial gradient over newly inserted blocks. The latter partial gradient
    is subsequently compared against the full gradient of entire circuit.

    Args:
        conf: configuration parameters, expects: "num_qubits" (int),
              "front_layer" (bool), "depth" (int), "entangler" ("cx", cz", cp").
        objv: class instance of objective function implementation.

    Returns:
        empty dictionary.
    """
    tol = float(np.sqrt(np.finfo(np.float64).eps))
    num_qubits = int(conf["num_qubits"])
    front_layer = bool(conf["front_layer"])

    # Create an ansatz and add extra unit blocks to the circuit and subsequently
    # compute the partial gradient over the newly added blocks.
    if objv.is_trotterized_ansatz():
        num_layers = int(np.random.randint(1, 3))
        blocks = make_trotter_like_circuit(num_qubits, num_layers)
        circ = TrotterAnsatz(num_qubits, blocks, second_order=objv.trotter_order())
        num_layers = int(np.random.randint(1, 3))
        new_blocks = make_trotter_like_circuit(num_qubits, num_layers)
        insert_pos = circ.bpl * np.random.randint(0, circ.num_layers + 1)
    else:
        depth = int(np.random.randint(num_qubits, 3 * num_qubits))
        blocks = tut.rand_circuit(num_qubits, depth)
        circ = ParametricCircuit(num_qubits, str(conf["entangler"]), blocks)
        depth = int(np.random.randint(num_qubits // 2, 2 * num_qubits))
        new_blocks = tut.rand_circuit(num_qubits, depth)
        insert_pos = np.random.randint(0, circ.num_blocks + 1)

    block_range = (insert_pos, insert_pos + new_blocks.shape[1])
    thetas = helper.rand_thetas(circ.num_thetas)
    thetas, idx = circ.insert_unit_blocks(insert_pos, new_blocks, thetas)
    if not np.all(thetas[idx] == 0):
        raise AssertionError("expects all zero newly added thetas")

    # Compute partial and full gradients.
    objv.initialize(circ=circ, thetas=thetas, close_states=False)
    g_full = objv.gradient(thetas)
    g_part = objv.gradient(thetas, block_range=block_range, front_layer=front_layer)

    # Depending on "front_layer" flag, either gradients of 1-qubit
    # front layer are all zeros or they coincide with full gradient.
    front_part, front_full = circ.subset1q(g_part), circ.subset1q(g_full)
    if front_layer:
        if not np.allclose(front_part, front_full, atol=tol, rtol=tol):
            raise AssertionError("mismatch in front layer gradients")
    else:
        if not np.allclose(front_part, 0, atol=tol, rtol=tol):
            raise AssertionError("expects zero front layer gradients")

    # Partial and full gradients agree on the newly added unit blocks.
    if not np.allclose(g_part[idx], g_full[idx], atol=tol, rtol=tol):
        raise AssertionError("gradient mismatch on newly added blocks")

    # Partial and full gradients agree on the newly added unit blocks;
    # another way to check consistency.
    new_part = circ.subset2q(g_part)[block_range[0] : block_range[1], :]
    new_full = circ.subset2q(g_full)[block_range[0] : block_range[1], :]
    if not np.allclose(new_part, new_full, atol=tol, rtol=tol):
        raise AssertionError("gradient mismatch on newly added blocks")

    # Gradients on the old unit blocks must be zero.
    old_part = circ.subset2q(g_part)[0 : block_range[0], :]
    if old_part.size > 0 and not np.allclose(old_part, 0, atol=tol, rtol=tol):
        raise AssertionError("expects zero gradients on the old blocks")
    old_part = circ.subset2q(g_part)[block_range[1] :, :]
    if old_part.size > 0 and not np.allclose(old_part, 0, atol=tol, rtol=tol):
        raise AssertionError("expects zero gradients on the old blocks")
    return dict({"trotterized": objv.is_trotterized_ansatz()})


def gradient_vs_numeric(conf: dict, objv: BaseGradTestObjective) -> dict:
    """
    Runs a single gradient test for test_dot_gradient_vs_numeric().
    """

    # Define the ansatz and its randomly selected state (thetas).
    num_qubits = int(conf["num_qubits"])
    if objv.is_trotterized_ansatz():
        num_layers = int(np.random.randint(1, 3))
        blocks = make_trotter_like_circuit(num_qubits, num_layers)
        second_order = objv.trotter_order()
        circ = TrotterAnsatz(num_qubits, blocks, second_order=second_order)
    else:
        depth = int(np.random.randint(num_qubits, 3 * num_qubits))
        blocks = create_ansatz_structure(num_qubits, "spin", "full", depth)
        second_order = "none"  # not-applicable
        circ = ParametricCircuit(num_qubits, str(conf["entangler"]), blocks)
    thetas = helper.rand_thetas(circ.num_thetas)

    # Compute objective function and its (exact) gradient.
    tol = float(np.sqrt(np.finfo(np.float64).eps))
    objv.initialize(circ=circ, thetas=thetas, close_states=True)
    fobj = objv.objective(thetas)
    tic = perf_counter()
    grad = objv.gradient(thetas)
    grad_time = perf_counter() - tic
    if not np.isclose(fobj, objv.objective_from_matrix(thetas), atol=tol, rtol=tol):
        print(abs(fobj), abs(objv.objective_from_matrix(thetas)))
        raise AssertionError("mismatch between objective function values")

    # Angles vary between -pi and pi. We start with reasonably small angle
    # increment (tau) and then gradually decrease it towards zero.
    tau = 0.25
    residual_prev = 1e20
    orders, errors = list([]), list([])
    for step in range(12):
        # Compute numerical gradient.
        num_grad = _numerical_gradient(objv.objective_from_matrix, thetas, tau)

        # Mean relative error of analytical gradient vs. numerical one.
        grad_norm = max(np.linalg.norm(grad), np.finfo(np.float64).eps ** 2)
        rel_err = np.linalg.norm(grad - num_grad) / grad_norm

        # Estimate the residual of Taylor expansion.
        grad_dir = grad / grad_norm
        delta = np.real(grad_dir * tau)
        fobj_delta = objv.objective(thetas + delta)
        residual = abs(fobj + np.dot(grad, delta) - fobj_delta)

        # Store relative gradient error and order of approximation
        # (it should be quadratic for the small steps).
        approx_order = (np.log(residual_prev) - np.log(residual)) / np.log(2.0)
        errors.append(float(rel_err))
        orders.append(float(0 if step == 0 else approx_order))

        # Proceed to the next iteration (more precise numerical gradient).
        tau /= 2
        residual_prev = residual

    return {
        "num_qubits": num_qubits,
        "depth": circ.num_blocks,
        "entangler": conf["entangler"],
        "num_thetas": circ.num_thetas,
        "grad_time": grad_time,
        "errors": errors,
        "orders": orders,
        "trotterized": objv.is_trotterized_ansatz(),
        "flip_bit": conf["flip_bit"],
        "trotter_order": (
            (2 if second_order else 1) if isinstance(second_order, bool) else second_order
        ),
    }


def make_configs(
    max_num_qubits: int,
    *,
    switch_front_layer: bool = False,
    entanglers: Optional[List[str]] = None,
    enable_flip_bit: bool = False,
) -> List[Dict]:
    """
    Generates configurations for all the simulations.

    Args:
        max_num_qubits: maximum number of qubits in test.
        switch_front_layer: enables tests with and without front layer,
                            default is with the front layer.
        entanglers: list of entanglers; a subset of ["cx", "cz", "cp"].
        enable_flip_bit: enable flipped bits in configurations.

    Returns:
        list of configurations for individual simulations.
    """
    assert chk.is_int(max_num_qubits, max_num_qubits >= 2)
    assert chk.is_bool(switch_front_layer)
    # Repeat "cx" for better coverage of Trotter case.
    entanglers = ["cx", "cx", "cx", "cz", "cp"] if entanglers is None else entanglers
    assert chk.is_list(entanglers) and set(entanglers).issubset(["cx", "cz", "cp"])
    return [
        {"num_qubits": n, "entangler": e, "front_layer": f, "flip_bit": b}
        for n in range(2, max_num_qubits + 1)
        for e in entanglers
        for f in ([False, True] if switch_front_layer else [True])
        for b in ([-1, np.random.randint(0, n)] if enable_flip_bit else [-1])
    ]


def aggregate_results(
    results: List[Dict],
    max_error: float = 1e-5,
    order_range: Tuple[float, float] = (1.8, 2.2),
):
    """
    Aggregates and verifies all the results.
    """
    assert isinstance(results, list) and isinstance(results[0], dict)
    assert isinstance(max_error, float) and 0 < max_error < 1
    assert isinstance(order_range, tuple) and len(order_range) == 2
    assert 1.5 <= order_range[0] < order_range[1] <= 2.5

    # Check correctness.
    for res in results:
        np_errors = np.array(res["errors"])
        np_orders = np.array(res["orders"])
        if res["status"] != "ok":
            raise AssertionError(f"failed result: {res['status']}")

        if not np.all(np_errors[-4:] <= 1e-5).item():
            raise AssertionError(f"residual exceeded the threshold: {max_error}")

        if not np.logical_and(
            np.all(order_range[0] <= np_orders[-4:]),
            np.all(np_orders[-4:] <= order_range[1]),
        ).item():
            raise AssertionError(
                f"Taylor expansion residual order fell outside admissible interval "
                f"[{order_range[0]}, {order_range[1]}]"
            )
