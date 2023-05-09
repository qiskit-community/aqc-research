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
Gradient test for the classical approach with the full state vectors,
where the objective function is defined as a dot product ``<x|V.H|y>``.
"""

import unittest
from typing import Optional, Tuple
import numpy as np
from qiskit.test import QiskitTestCase
from aqc_research.parametric_circuit import ParametricCircuit
import test.utils_for_testing as tut
import aqc_research.core_operations as cop
from aqc_research.job_executor import run_jobs
import aqc_research.circuit_transform as ctr
import test.utils_dot_gradient_test as gradtest
import aqc_research.checking as chk


class _Objective(gradtest.BaseGradTestObjective):
    def __init__(self, can_trotterize: bool):
        assert isinstance(can_trotterize, bool)
        self.circ = None
        self.x_vec = None
        self.y_vec = None
        self.workspace = None
        self.trotterized = can_trotterize and bool(np.random.rand() >= 0.34)
        self.second_order = can_trotterize and bool(np.random.rand() >= 0.6)

    def is_trotterized_ansatz(self) -> bool:
        return self.trotterized

    def trotter_order(self) -> bool:
        return self.second_order

    def initialize(self, circ: ParametricCircuit, thetas: np.ndarray, close_states: bool):
        assert isinstance(circ, ParametricCircuit)
        assert chk.float_1d(thetas, thetas.size == circ.num_thetas)
        assert isinstance(close_states, bool)

        self.circ = circ
        self.workspace = np.zeros((3, circ.dimension), dtype=np.cfloat)
        if close_states:  # x and y vectors are close to each other
            self.x_vec = tut.rand_vec(circ.dimension, True)
            self.y_vec = self.x_vec + 0.1 * tut.rand_vec(circ.dimension, True)
        else:  # x and y vector are completely independent
            self.x_vec = tut.rand_vec(circ.dimension, True)
            self.y_vec = tut.rand_vec(circ.dimension, True)

    def objective_from_matrix(self, thetas: np.ndarray) -> np.cfloat:
        v_mat = ctr.qcircuit_to_matrix(ctr.ansatz_to_qcircuit(self.circ, thetas))
        return np.cfloat(np.vdot(v_mat @ self.x_vec, self.y_vec))

    def objective(self, thetas: np.ndarray) -> np.cfloat:
        vh_y = np.zeros(self.circ.dimension, dtype=np.cfloat)
        vh_y = cop.v_dagger_mul_vec(
            circ=self.circ,
            thetas=thetas,
            vec=self.y_vec,
            out=vh_y,
            workspace=self.workspace[0:2],
        )
        return np.cfloat(np.vdot(self.x_vec, vh_y))

    def gradient(
        self,
        thetas: np.ndarray,
        block_range: Optional[Tuple[int, int]] = None,
        front_layer: Optional[bool] = True,
    ) -> np.ndarray:
        vh_y = np.zeros(self.circ.dimension, dtype=np.cfloat)
        vh_y = cop.v_dagger_mul_vec(
            circ=self.circ,
            thetas=thetas,
            vec=self.y_vec,
            out=vh_y,
            workspace=self.workspace[0:2],
        )
        return cop.grad_of_dot_product(
            circ=self.circ,
            thetas=thetas,
            x_vec=self.x_vec,
            vh_y_vec=vh_y,
            workspace=self.workspace,
            block_range=block_range,
            front_layer=front_layer,
        )


class TestDotGradient(QiskitTestCase):
    """
    Compares full gradient of the dot product ``<x|V.H|y>`` against numeric one.
    Compares partial vs full gradient.
    """

    _max_num_qubits = 3  # max. number of qubits
    _seed = 0x696969  # seed for random generator

    def setUp(self):
        super().setUp()
        np.random.seed(self._seed)

    def test_partial_vs_full_grad(self):
        """Tests correctness of partial gradient vs full one."""

        def _job_function(_: int, _conf: dict) -> dict:
            _objv = _Objective(can_trotterize=bool(_conf["entangler"] == "cx"))
            return gradtest.partial_vs_full_gradient_test(_conf, _objv)

        configs = gradtest.make_configs(
            self._max_num_qubits,
            switch_front_layer=True,
            entanglers=["cx", "cx", "cx", "cz", "cp"],  # repeats "cx" for Trotter
        )
        results = run_jobs(configs, self._seed, _job_function)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    def test_dot_gradient_vs_numeric(self):
        """Tests gradient vs numerical one."""

        def _job_function(_: int, _conf: dict) -> dict:
            _objv = _Objective(can_trotterize=bool(_conf["entangler"] == "cx"))
            return gradtest.gradient_vs_numeric(_conf, _objv)

        configs = gradtest.make_configs(
            self._max_num_qubits,
            entanglers=["cx", "cx", "cx", "cz", "cp"],  # repeats "cx" for Trotter
        )
        results = run_jobs(configs, self._seed, _job_function)
        gradtest.aggregate_results(results)


if __name__ == "__main__":
    unittest.main()
