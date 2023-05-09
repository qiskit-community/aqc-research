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
Tests for the module 'model_sp_lhs/objective_base.py'.
"""

import unittest
import numpy as np
from qiskit.test import QiskitTestCase
from qiskit import QuantumCircuit
from aqc_research.model_sp_lhs.objective_base import ThinStateHandler
from aqc_research.circuit_transform import qcircuit_to_state
import aqc_research.core_operations as cop


class TestSPObjectiveBase(QiskitTestCase):
    """Tests for the module 'model_sp_lhs/objective_base.py'."""

    max_num_qubits = 5  # maximum number of qubits in tests
    num_repeats = 50  # number of repetitions in tests

    def setUp(self):
        super().setUp()
        np.random.seed(0x0696969)

    def test_state_handler(self):
        """
        Tests the ThinStateHandler class.
        """
        tol = float(np.sqrt(np.finfo(np.float64).eps))
        for n_q in range(2, self.max_num_qubits + 1):
            qiskit_convention = bool(cop.bit2bit_transform(n_q, 0) != 0)

            for max_flips in range(1, n_q + 1):
                tsh = ThinStateHandler(n_q, max_flips)
                num_states = tsh.num_states
                comb_labels = tsh.flip_qubit_positions

                # Generate the states |0>, X_j@|0>, X_i@X_j@|0>, etc. in a
                # straightforward way and compare them to ThinStateHandler's ones.
                zero = np.array([[1], [0]], dtype=np.int64)  # single qubit |0>
                unit = np.array([[0], [1]], dtype=np.int64)  # single qubit |1>
                bits = np.zeros(n_q, dtype=np.int8)

                # Check the state |0>.
                sh_state = tsh.init_state(0)
                self.assertTrue(sh_state.dtype == np.cfloat)
                self.assertTrue(np.allclose(sh_state[0], 1, atol=tol, rtol=tol))
                self.assertTrue(np.allclose(sh_state[1:], 0, atol=tol, rtol=tol))

                # Check all flip-states.
                count = 1
                for flips in range(max_flips):
                    for j in range(len(comb_labels[flips])):
                        # Flip bits of initially zero state |00..0>.
                        bits.fill(0)
                        for i in range(flips + 1):
                            k = comb_labels[flips][j][i]
                            self.assertTrue(bits[k] == 0)
                            bits[k] = 1

                        # Straightforwardly assemble state with flipped bits.
                        state = np.int64(1)
                        for i in range(n_q):
                            if qiskit_convention:
                                state = np.kron(unit if bits[i] != 0 else zero, state)
                            else:
                                state = np.kron(state, unit if bits[i] != 0 else zero)
                        state = state.ravel()
                        self.assertTrue(np.count_nonzero(state) == 1)

                        # Assemble state from quantum circuit.
                        qcirc = QuantumCircuit(n_q)
                        for i in range(flips + 1):
                            pos = comb_labels[flips][j][i]
                            qcirc.x(pos)
                        qc_state = qcircuit_to_state(qcirc)

                        # Compare alternative implementations against ThinStateHandler.
                        sh_state = tsh.init_state(count)
                        self.assertTrue(np.allclose(sh_state, state, atol=tol, rtol=tol))
                        # Note, this will fail in case of non-Qiskit bit ordering.
                        self.assertTrue(
                            np.allclose(qc_state, state, atol=tol, rtol=tol),
                            "check that you follow Qiskit bit ordering",
                        )

                        count += 1
                self.assertTrue(count == num_states)


if __name__ == "__main__":
    unittest.main()
