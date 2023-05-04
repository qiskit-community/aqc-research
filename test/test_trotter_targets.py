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
Tests correctness of utilities for target preparation in target_states.py.
"""

import unittest
from dataclasses import dataclass
import numpy as np
from qiskit.test import QiskitTestCase
import aqc_research.utils as helper
import aqc_research.model_sp_lhs.trotter.trotter as trotop
import aqc_research.mps_operations as mpsop
import aqc_research.model_sp_lhs.trotter.target_states as trotst


@dataclass(frozen=True)
class _Options:
    """Imitates user options."""

    delta = 1.0  # parameter "delta" in Hamiltonian
    ini_state_func = (trotop.neel_init_state,)  # initial state generator
    trunc_thr_target = mpsop.no_truncation_threshold()  # truncation threshold
    _big_step = float(1.2)
    _num_big_time_steps = int(5)
    trotter_steps = (1 + np.arange(_num_big_time_steps)) * round(_big_step / 0.4)
    evol_times = np.round((1 + np.arange(_num_big_time_steps)) * _big_step, 3)


class TestTrotterTargets(QiskitTestCase):
    """
    Tests correctness of utilities for target preparation in target_states.py.
    """

    _max_num_qubits = 5  # max. number of qubits
    _seed = 0x696969  # seed for random generator

    def setUp(self):
        super().setUp()
        np.random.seed(self._seed)

    def test_generate_all_mps_targets(self):
        """Tests generate_all_mps_targets() function."""
        for num_qubits in range(2, self._max_num_qubits + 1):
            hamiltonian = trotop.make_hamiltonian(num_qubits, _Options.delta)
            for second_order in [False, True]:
                targets = trotst.generate_all_mps_targets(
                    opts=_Options(), num_qubits=num_qubits, second_order=second_order
                )
                for targ in targets:
                    timer = helper.MyTimer()

                    # By default, we set Trotter global phase to 0 (because it
                    # is difficult to keep track of phase factor everywhere);
                    # instead, we compensate the global phase of the exact
                    # state by the inverse phase factor.
                    with timer("exact_state"):
                        exact_state = trotop.exact_evolution(
                            hamiltonian=hamiltonian,
                            ini_state=_Options.ini_state_func[0](num_qubits),
                            evol_time=targ.evol_time,
                        )
                        nsteps = targ.num_trot_steps * trotst.precise_multiplier()
                        exact_state *= np.exp(
                            -1j * trotop.trotter_global_phase(num_qubits, nsteps, second_order)
                        )

                    with timer("t1_gt_to_state"):
                        t1_gt_state = mpsop.mps_to_vector(targ.t1_gt)

                    with timer("t1_to_state"):
                        t1_state = mpsop.mps_to_vector(targ.t1)

                    diff_gt = trotop.state_difference(exact_state, t1_gt_state)
                    fid_gt = trotop.fidelity(exact_state, t1_gt_state)
                    fid = trotop.fidelity(exact_state, t1_state)

                    if not second_order:
                        self.assertTrue(diff_gt < 0.03)

                    self.assertTrue(fid_gt > 0.99)
                    self.assertTrue(fid > 0.93)


if __name__ == "__main__":
    unittest.main()
