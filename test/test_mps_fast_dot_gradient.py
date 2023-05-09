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
Gradient test for the accelerated (aka "fast") MPS-based approach with state
vectors in MPS format, where the objective function is defined as a dot product
``<0|V.H|phi>``, where ``V`` is the matrix of ansatz.
"""

import unittest
from typing import Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from aqc_research.parametric_circuit import ParametricCircuit
import aqc_research.utils as helper
import aqc_research.mps_operations as mpsop
import aqc_research.mps_dot_objective as mpsobj
from aqc_research.circuit_transform import qcircuit_to_matrix, ansatz_to_qcircuit
from aqc_research.job_executor import run_jobs
import aqc_research.checking as chk
import test.utils_dot_gradient_test as gradtest


class _Objective(gradtest.BaseGradTestObjective):
    """A simple objective function for testing."""

    def __init__(self, can_trotterize: bool):
        self.circ = None
        self.lvec_mps = None  # left-hand side vector |0> in MPS format
        self.lvec_state = None  # left-hand side vector |0> as a normal vector
        self.target_mps = None  # vector |phi> in MPS format
        self.target_state = None  # vector |phi> as a normal vector
        self.trunc_thr = mpsop.no_truncation_threshold()
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

        # Create left-hand side vector as MPS and normal states.
        qcirc = QuantumCircuit(circ.num_qubits)
        self.lvec_state = np.zeros(circ.dimension, dtype=np.cfloat)
        self.lvec_mps = mpsop.mps_from_circuit(
            qcirc, trunc_thr=self.trunc_thr, out_state=self.lvec_state
        )
        self.circ = circ

        # Create right-hand side vector (in MPS format and as a normal state).
        target_thetas = helper.rand_thetas(circ.num_thetas)
        if close_states:
            target_thetas = thetas + 0.1 * target_thetas
        qcirc = ansatz_to_qcircuit(circ, target_thetas)
        self.target_state = np.zeros(circ.dimension, dtype=np.cfloat)
        self.target_mps = mpsop.mps_from_circuit(
            qcirc, trunc_thr=self.trunc_thr, out_state=self.target_state
        )

    def objective_from_matrix(self, thetas: np.ndarray) -> np.cfloat:
        v_mat = qcircuit_to_matrix(ansatz_to_qcircuit(self.circ, thetas))
        return np.cfloat(np.vdot(v_mat @ self.lvec_state, self.target_state))

    def objective(self, thetas: np.ndarray) -> np.cfloat:
        vh_phi = mpsop.v_dagger_mul_mps(
            self.circ,
            thetas,
            self.target_mps,
            trunc_thr=self.trunc_thr,
        )
        return mpsop.mps_dot(self.lvec_mps, vh_phi)

    def gradient(
        self,
        thetas: np.ndarray,
        block_range: Optional[Tuple[int, int]] = None,
        front_layer: Optional[bool] = True,
    ) -> np.ndarray:
        vh_phi = mpsop.v_dagger_mul_mps(
            self.circ,
            thetas,
            self.target_mps,
            trunc_thr=self.trunc_thr,
        )
        return mpsobj.fast_dot_gradient(
            circ=self.circ,
            thetas=thetas,
            lvec=self.lvec_mps,
            vh_phi=vh_phi,
            trunc_thr=self.trunc_thr,
            block_range=block_range,
            front_layer=front_layer,
        )


class TestMPSFastDotGradient(QiskitTestCase):
    """
    Compares full gradient of the dot product ``<0|V.H|phi>`` against numeric
    one, where state vectors are represented in MPS format.
    Compares partial vs full gradient, also in MPS representation.
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
