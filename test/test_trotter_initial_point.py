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
Tests correctness of utilities defined in trotter_initial_point.py.
"""

import unittest
from typing import Tuple
from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
import numpy as np
import aqc_research.utils as helper
from aqc_research.circuit_structures import make_trotter_like_circuit
from aqc_research.circuit_transform import qcircuit_to_matrix, ansatz_to_qcircuit
from aqc_research.parametric_circuit import TrotterAnsatz
import aqc_research.model_sp_lhs.trotter.trotter as trotop
import test.utils_for_testing as tut
from aqc_research.job_executor import run_jobs


class TestTrotterInitialPoint(QiskitTestCase):
    """
    Tests correctness of utilities defined in trotter_initial_point.py.
    """

    _max_num_qubits = 5  # max. number of qubits
    _max_num_steps = 4  # max. number of Trotter steps
    _num_repeats = 2  # number of test repetitions
    _seed = 0x696969  # seed for random generator

    def setUp(self):
        super().setUp()
        np.random.seed(self._seed)

    @staticmethod
    def _rand_range(num_steps: int) -> Tuple[int, int]:
        """Selects a random sub-range of Trotter steps."""
        low, high = 0, 0
        while low == high:
            rng = np.random.randint(0, num_steps + 1, size=2)
            low, high = np.amin(rng), np.amax(rng)
        return low, high

    def _job_init_ansatz_to_trotter(self, _: int, conf: dict) -> dict:
        """Job function for test_init_ansatz_to_trotter()."""
        tol = np.sqrt(np.finfo(np.float64).eps)
        num_qubits = int(conf["num_qubits"])
        num_steps = int(conf["num_steps"])
        second_order = bool(conf["second_order"])
        delta_t = 1.0

        # Genuine Trotter.
        qc1 = trotop.trotter_circuit(
            QuantumCircuit(num_qubits),
            dt=delta_t,
            delta=1.0,
            num_trotter_steps=num_steps,
            second_order=second_order,
        )

        # Trotter from ansatz, all layers are initialized.
        blocks = make_trotter_like_circuit(num_qubits, num_layers=num_steps, verbose=False)
        circ = TrotterAnsatz(num_qubits, blocks, second_order)
        evol_time = delta_t * circ.num_layers
        thetas = helper.rand_thetas(circ.num_thetas)
        thetas = trotop.init_ansatz_to_trotter(
            circ, thetas, evol_time=evol_time, delta=1.0, layer_range=None
        )
        qc2 = ansatz_to_qcircuit(circ, thetas)

        # Trotter from ansatz, but only sub-range of layers is initialized,
        # other ones remain the same.
        layer_range = self._rand_range(num_steps)
        evol_time = delta_t * (layer_range[1] - layer_range[0])
        thetas = trotop.init_ansatz_to_trotter(
            circ, thetas, evol_time=evol_time, delta=1.0, layer_range=layer_range
        )
        qc3 = ansatz_to_qcircuit(circ, thetas)

        self.assertTrue(np.all(np.isclose(qc1.global_phase, qc2.global_phase)))
        self.assertTrue(np.all(np.isclose(qc1.global_phase, qc3.global_phase)))
        mat1 = qcircuit_to_matrix(qc1)
        mat2 = qcircuit_to_matrix(qc2)
        mat3 = qcircuit_to_matrix(qc3)
        self.assertTrue(tut.relative_diff(mat2, mat1) < tol)
        self.assertTrue(tut.relative_diff(mat3, mat1) < tol)
        return dict({})

    def test_init_ansatz_to_trotter(self):
        """Tests trotop.init_ansatz_to_trotter() function."""
        configs = [
            {"num_qubits": n, "num_steps": s, "second_order": o}
            for _ in range(self._num_repeats)
            for n in range(2, self._max_num_qubits + 1)
            for s in range(1, self._max_num_steps + 1)
            for o in [False, True]
        ]
        results = run_jobs(configs, self._seed, self._job_init_ansatz_to_trotter)
        self.assertTrue(all(r["status"] == "ok" for r in results))


if __name__ == "__main__":
    unittest.main()
