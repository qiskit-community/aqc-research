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
Tests for the module 'circuit_transform.py'.
"""

import unittest
from time import perf_counter
import numpy as np
from aqc_research.parametric_circuit import ParametricCircuit, TrotterAnsatz
import aqc_research.utils as helper
import test.utils_for_testing as tut
import aqc_research.circuit_transform as ctr
from aqc_research.job_executor import run_jobs
from aqc_research.circuit_structures import make_trotter_like_circuit


class TestCircuitTransforms(unittest.TestCase):
    """Tests for the module 'circuit_transform.py'."""

    _max_num_qubits = 5  # maximum number of qubits in tests
    _num_repeats = 3  # number of repetitions in tests
    _seed = 0x696969  # seed for random generator

    def setUp(self):
        super().setUp()
        np.random.seed(self._seed)

    def _job_cmp_circuit_matrices(self, _: int, config: dict) -> dict:
        """Runs test for a single combination of #qubits and entangler type."""
        tol = float(np.sqrt(np.finfo(np.float64).eps))
        num_qubits = config["num_qubits"]

        if config["entangler"] == "cx" and bool(np.random.rand() >= 0.5):
            blocks = make_trotter_like_circuit(num_qubits, np.random.randint(1, 4))
            second_order = bool(np.random.rand() >= 0.6)
            circ = TrotterAnsatz(num_qubits, blocks, second_order=second_order)
        else:
            blocks = tut.rand_circuit(num_qubits, np.random.randint(10, 20))
            circ = ParametricCircuit(num_qubits, config["entangler"], blocks)
        thetas = helper.rand_thetas(circ.num_thetas)
        metrics = dict({})

        tic = perf_counter()
        mat1 = ctr.ansatz_to_numpy_trotter(circ, thetas)
        metrics["numpy"] = perf_counter() - tic

        tic = perf_counter()
        mat2 = ctr.ansatz_to_numpy_by_qiskit(circ, thetas)
        metrics["qiskit"] = perf_counter() - tic

        # ansatz_to_numpy() does not support Trotterized ansatz.
        tic = perf_counter()
        if isinstance(circ, TrotterAnsatz):
            mat3 = mat1
        else:
            mat3 = ctr.ansatz_to_numpy_fast(circ, thetas)
        metrics["fast_numpy"] = perf_counter() - tic

        self.assertTrue(np.allclose(mat1, mat2, atol=tol, rtol=tol))
        self.assertTrue(np.allclose(mat1, mat3, atol=tol, rtol=tol))

        stv1 = mat1 @ helper.zero_state(circ.num_qubits)
        stv2 = ctr.qcircuit_to_state(ctr.ansatz_to_qcircuit(circ, thetas))
        self.assertTrue(np.allclose(stv1, stv2, atol=tol, rtol=tol))
        return metrics

    def test_cmp_circuit_matrices(self):
        """
        Test compares circuit matrices V(Theta) constructed by
        Numpy and Qiskit implementations.
        """
        configs = [
            {"num_qubits": n, "entangler": e}
            for n in range(2, self._max_num_qubits + 1)
            for e in ["cx", "cx", "cx", "cz", "cp"]  # repeats "cx" for Trotter
            for _ in range(self._num_repeats)
        ]
        results = run_jobs(configs, self._seed, self._job_cmp_circuit_matrices)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    def test_insert_unit_blocks(self):
        """
        Tests the function insert_unit_blocks() in parametric circuit class.
        """
        for num_qubits in range(2, self._max_num_qubits + 1):
            blocks = helper.rand_circuit(num_qubits, np.random.randint(10, 20))
            circ = ParametricCircuit(num_qubits, "cx", blocks)
            thetas = np.arange(circ.num_thetas).astype(float)
            for depth in [int(0), int(np.random.randint(10, 20))]:
                blocks_ex = helper.rand_circuit(num_qubits, depth)
                for pos in range(circ.num_blocks + 1):
                    circ_ex = ParametricCircuit(num_qubits, "cx", blocks.copy())
                    thetas_ex, idx = circ_ex.insert_unit_blocks(pos, blocks_ex, thetas.copy())
                    vals_ex = 111 * np.arange(1, idx.size + 1)
                    thetas_ex[idx] = vals_ex

                    # Front gates' thetas remain the same.
                    th1, th2 = circ.subset1q(thetas), circ_ex.subset1q(thetas_ex)
                    self.assertTrue(np.all(th1 == th2))

                    # 2-qubits blocks' thetas before insertion point remain the same.
                    th1, th2 = circ.subset2q(thetas), circ_ex.subset2q(thetas_ex)
                    self.assertTrue(np.all(th1[0:pos] == th2[0:pos]))

                    # Newly inserted blocks take their values.
                    num_ex = blocks_ex.shape[1]
                    self.assertTrue(np.all(th2[pos : pos + num_ex].ravel() == vals_ex))

                    # 2-qubits blocks' thetas after inserted blocks remain the same.
                    self.assertTrue(np.all(th1[pos:] == th2[(pos + num_ex) :]))


if __name__ == "__main__":
    unittest.main()
