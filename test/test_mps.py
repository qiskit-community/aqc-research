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
Tests correctness of our interpretation of Qiskit MPS implementation.
"""

from time import perf_counter
import unittest
from typing import List, Dict
from qiskit.test import QiskitTestCase
import numpy as np
from aqc_research.parametric_circuit import ParametricCircuit
import test.utils_for_testing as tut
from aqc_research.circuit_structures import create_ansatz_structure
from aqc_research.circuit_transform import ansatz_to_qcircuit
import aqc_research.utils as helper
import aqc_research.mps_operations as mpsop
from aqc_research.job_executor import run_jobs
import aqc_research.circuit_transform as ctr


class TestMPS(QiskitTestCase):
    """
    Tests correctness of our interpretation of Qiskit MPS implementation.
    **Note**, MPS is defined up to global phase factor (after transpiler?).
    Often the phase factor can be safely dropped, but not always.
    """

    _max_num_qubits = 7  # max. number of qubits
    _num_repeats = 4  # number of test repetitions per qubit number
    _seed = 0x696969  # seed for random generator
    _tol = float(np.sqrt(np.finfo(float).eps))  # tolerance

    def setUp(self):
        super().setUp()
        np.random.seed(self._seed)

    def _make_configs(self) -> List[Dict]:
        """Generates configurations for all the simulations."""
        return [
            {"num_qubits": n, "entangler": e}
            for n in range(2, self._max_num_qubits + 1)
            for e in ["cx", "cz", "cp"]
            for _ in range(self._num_repeats)
        ]

    def _job_mps_to_vector(self, _: int, conf: dict) -> dict:
        """Job function for the test_mps_to_vector()."""
        num_qubits = int(conf["num_qubits"])
        tol = (2 ** max(num_qubits - 10, 0)) * self._tol

        # Generates a random state in MPS format.
        state1 = np.zeros(2**num_qubits, dtype=np.cfloat)
        mps = mpsop.rand_mps_vec(num_qubits, out_state=state1)
        self.assertTrue(mpsop.check_mps(mps))

        # Reconstructed MPS vector must coincide with the one returned by Qiskit.
        tic = perf_counter()
        state2 = mpsop.mps_to_vector(mps)
        clock = perf_counter() - tic
        residual = tut.relative_diff(state1, state2)  # note, phase == 0
        self.assertTrue(residual < tol, f"too large residual: {residual}")
        return {"num_qubits": num_qubits, "residual": residual, "exec_time": clock}

    def test_mps_to_vector(self):
        """Tests the function mpsop.mps_to_vector()."""
        results = run_jobs(self._make_configs(), self._seed, self._job_mps_to_vector)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    def _job_mps_dot(self, _: int, conf: dict) -> dict:
        """Job function for the test_mps_dot()."""
        num_qubits = int(conf["num_qubits"])
        tol = (2 ** max(num_qubits - 10, 0)) * self._tol
        state1 = np.zeros(2**num_qubits, dtype=np.cfloat)
        state2 = np.zeros(2**num_qubits, dtype=np.cfloat)

        mps1 = mpsop.rand_mps_vec(num_qubits, out_state=state1)
        mps2 = mpsop.rand_mps_vec(num_qubits, out_state=state2)

        tic = perf_counter()
        dot11 = mpsop.mps_dot(mps1, mps1)
        dot12 = mpsop.mps_dot(mps1, mps2)
        dot22 = mpsop.mps_dot(mps2, mps2)
        clock = float(perf_counter() - tic) / 3

        err11 = abs(dot11 - np.cfloat(np.vdot(state1, state1)))
        err12 = abs(dot12 - np.cfloat(np.vdot(state1, state2)))
        err22 = abs(dot22 - np.cfloat(np.vdot(state2, state2)))

        err11 = max(abs(dot11 - 1), err11)
        err22 = max(abs(dot22 - 1), err22)

        for err, kind in zip([err11, err12, err22], ["11", "12", "22"]):
            self.assertTrue(err < tol, f"too large residual{kind}: {err}")

        residual = max(err11, err12, err22)
        return {"num_qubits": num_qubits, "residual": residual, "exec_time": clock}

    def test_mps_dot(self):
        """Tests the function mpsop.mps_dot()."""
        results = run_jobs(self._make_configs(), self._seed, self._job_mps_dot)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    def _job_qcircuit_mul_mps(self, _: int, conf: dict) -> dict:
        """Job function for the test_qcircuit_mul_mps()."""
        tol = 2 * self._tol
        num_qubits = int(conf["num_qubits"])

        # Create 2 circuits.
        qcirc = list([])
        for _ in range(2):
            blocks = create_ansatz_structure(num_qubits, "spin", "full", 3 * (num_qubits - 1))
            circ = ParametricCircuit(num_qubits, conf["entangler"], blocks)
            thetas = helper.rand_thetas(circ.num_thetas)
            qcirc.append(ansatz_to_qcircuit(circ, thetas))

        # **Recall**, one can get state vector from circuit only once!
        # This is Qiskit limitation, so we create a copy of the circuit.

        # state1 <-- MPS(qc1) * MPS(qc0).
        mps0 = mpsop.mps_from_circuit(qcirc[0].copy())  # copy!
        state1 = np.zeros(2**num_qubits, dtype=np.cfloat)
        tic = perf_counter()
        mpsop.qcircuit_mul_mps(qcirc[1].copy(), mps0, out_state=state1)  # copy!
        clock = perf_counter() - tic

        # state2 <-- qc1 * qc0, state after concatenated circuits.
        qc2 = qcirc[0].compose(qcirc[1])
        state2 = ctr.qcircuit_to_state(qc2)

        residual = np.linalg.norm(state1 - state2)
        self.assertTrue(np.isclose(residual, 0, atol=tol, rtol=tol), f"residual: {residual}")
        return {"num_qubits": num_qubits, "residual": residual, "exec_time": clock}

    def test_qcircuit_mul_mps(self):
        """Tests the function mpsop.qcircuit_mul_mps()."""
        configs = self._make_configs()
        results = run_jobs(configs, self._seed, self._job_qcircuit_mul_mps)
        self.assertTrue(all(r["status"] == "ok" for r in results))

    def _job_v_mul_vec(self, _: int, conf: dict) -> dict:
        """Job function for the test_v_mul_vec()."""
        tol = self._tol
        residual = 0.0
        num_qubits = int(conf["num_qubits"])
        entangler = conf["entangler"]

        # Generate a random circuit and full-size vector.
        blocks = tut.rand_circuit(num_qubits, np.random.randint(20, 50))
        circ = ParametricCircuit(num_qubits, entangler, blocks)
        thetas = helper.rand_thetas(circ.num_thetas)
        vec = np.zeros(2**num_qubits, dtype=np.cfloat)
        mps_vec = mpsop.rand_mps_vec(num_qubits, out_state=vec)

        # vec2 = V @ V.H @ vec == vec.
        vec1 = mpsop.v_dagger_mul_mps(circ, thetas, mps_vec)
        vec2 = mpsop.v_mul_mps(circ, thetas, vec1)
        dot = mpsop.mps_dot(vec2, mps_vec)
        residual = max(residual, abs(dot - 1))
        self.assertTrue(np.isclose(dot, 1, atol=tol, rtol=tol))

        # MPS(V.H @ vec) == V.H @ vec, by converting from MPS to normal vector.
        qcirc = ctr.ansatz_to_qcircuit(circ, thetas)
        vh_mat = ctr.qcircuit_to_matrix(qcirc.inverse())
        dot = np.vdot(mpsop.mps_to_vector(vec1), vh_mat @ vec)
        residual = max(residual, abs(dot - 1))
        self.assertTrue(np.isclose(dot, 1, atol=tol, rtol=tol))

        # vec2 = V.H @ V @ vec == vec.
        vec1 = mpsop.v_mul_mps(circ, thetas, mps_vec)
        vec2 = mpsop.v_dagger_mul_mps(circ, thetas, vec1)
        dot = mpsop.mps_dot(vec2, mps_vec)
        residual = max(residual, abs(dot - 1))
        self.assertTrue(np.isclose(dot, 1, atol=tol, rtol=tol))

        # MPS(V @ vec) == V @ vec, by converting from MPS to normal vector.
        qcirc = ctr.ansatz_to_qcircuit(circ, thetas)
        v_mat = ctr.qcircuit_to_matrix(qcirc)
        dot = np.vdot(mpsop.mps_to_vector(vec1), v_mat @ vec)
        residual = max(residual, abs(dot - 1))
        self.assertTrue(np.isclose(dot, 1, atol=tol, rtol=tol))

        result = dict({})
        return result

    def test_v_mul_vec(self):
        """Tests the functions mpsop.v_mul_vec() and mpsop.v_dagger_mul_vec()."""
        results = run_jobs(self._make_configs(), self._seed, self._job_v_mul_vec)
        self.assertTrue(all(r["status"] == "ok" for r in results))


if __name__ == "__main__":
    unittest.main()
