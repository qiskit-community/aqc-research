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
Tests correctness of Trotter framework implementation.
"""

import unittest
from qiskit.test import QiskitTestCase
import numpy as np
import aqc_research.utils as helper
import aqc_research.model_sp_lhs.trotter.trotter as trotop
from aqc_research.mps_operations import mps_to_vector
from aqc_research.circuit_transform import qcircuit_to_state


class TestTrotterFramework(QiskitTestCase):
    """
    Tests correctness of Trotter framework implementation.
    """

    _max_num_qubits = 5  # max. number of qubits
    _seed = 0x696969  # seed for random generator
    _mps_trunc_thr = 1e-6  # MPS truncation threshold

    def setUp(self):
        super().setUp()
        np.random.seed(self._seed)

    def test_trotter_vs_exact(self):
        """Tests time evolution routines Trotter vs exact."""
        nsteps, delta = 30, 1.0
        for num_qubits in range(2, self._max_num_qubits + 1):
            hamiltonian = trotop.make_hamiltonian(num_qubits, delta)
            for second_order in [False, True]:
                for evol_tm in [0.5, 0.7, 1.0, 1.5, 2.0]:
                    ini_state_func = trotop.neel_init_state
                    timer = helper.MyTimer()

                    # Exact evolution for the full time interval.
                    # By default, we set Trotter global phase to 0 (because it
                    # is difficult to keep track of phase factor everywhere);
                    # instead, we compensate the global phase of the exact
                    # state by the inverse phase factor.
                    with timer("exact"):
                        exact_state = trotop.exact_evolution(
                            hamiltonian, ini_state_func(num_qubits), evol_tm
                        )
                        exact_state *= np.exp(
                            -1j * trotop.trotter_global_phase(num_qubits, nsteps, second_order)
                        )

                    # Apply Trotter twice over the half-time intervals.
                    with timer("trotter"):
                        half_trot1 = trotop.Trotter(
                            num_qubits=num_qubits,
                            evol_time=evol_tm * 0.5,
                            num_steps=nsteps // 2,
                            delta=delta,
                            second_order=second_order,
                        )
                        half_trot2 = trotop.Trotter(
                            num_qubits=num_qubits,
                            evol_time=evol_tm * 0.5,
                            num_steps=nsteps - nsteps // 2,
                            delta=delta,
                            second_order=second_order,
                        )
                        trot_state = half_trot2.as_qcircuit(
                            half_trot1.as_qcircuit(ini_state_func(num_qubits))
                        )
                        trot_state = qcircuit_to_state(trot_state)

                    # MPS evolution for the full time interval.
                    with timer("mps"):
                        full_trot = trotop.Trotter(
                            num_qubits=num_qubits,
                            evol_time=evol_tm,
                            num_steps=nsteps,
                            delta=delta,
                            second_order=second_order,
                        )
                        mps = full_trot.as_mps(
                            ini_state_func(num_qubits), trunc_thr=self._mps_trunc_thr
                        )

                    fid = trotop.fidelity(trot_state, exact_state)
                    mps_fid = trotop.fidelity(trot_state, mps_to_vector(mps))
                    self.assertTrue(fid > 0.9)
                    self.assertTrue(mps_fid > 0.9)


if __name__ == "__main__":
    unittest.main()
