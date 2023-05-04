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
Class keeps all user-supplied settings.
"""

import os
import time
from typing import Any, Optional
import numpy as np
import aqc_research.model_sp_lhs.trotter.trotter as trotop
from aqc_research.mps_operations import no_truncation_threshold


class UserOptions:
    """Class keeps all user-supplied settings."""

    def __init__(self, cargs: Optional[Any] = None):
        """
        Args:
            cargs: command-line argments; None implies default settings.
        """

        # The number of qubits in simulation, n >= 2.
        self.num_qubits = int(cargs.num_qubits) if cargs else int(5)

        # Pre-compute the target states and exit, if True.
        self.target_only = bool(cargs.target_only) if cargs else False

        # Tag-string that helps to identify the simulation results.
        self.tag = str(cargs.tag) if cargs else ""

        # The file to load the pre-computed target states from; None or empty
        # string implies default file path inside 'self.result_dir' folder.
        self.targets_file = str(cargs.targets_file) if cargs else ""

        # The output folder of simulation results.
        self.result_dir = os.path.join(os.getcwd(), "results", "trotter_evol")

        # Parameter "delta" in Hamiltonian - scale of z-terms.
        self.delta = float(1.0)

        # MPS truncation thresholds for computation with moderate accuracy and
        # for computation of the ground-truth target states with high accuracy.
        self.trunc_thr = float(1e-6)
        self.trunc_thr_target = no_truncation_threshold()

        # We move forward by the big time steps (which define the "time horizons").
        # New targets states ("ground-truth" and "reference" ones) are computed
        # after every big step. In between, the state propagation is carried
        # out by a small Trotter time step (dt), where, by definition, a small
        # time step corresponds to a single layer of 2-qubit elementary Trotter
        # blocks applied to every pair of adjacent qubits. In case of 2nd order
        # Trotter, a trailing half-layer is also implied.
        small_step = float(0.4)
        big_step = float(1.2)
        num_big_steps = int(6)
        step_range = 1 + np.arange(num_big_steps)

        # Parameter "trotter_steps" gives the number of Trotter layers
        # for all time horizons for computing the reference target state
        # with moderate accuracy. More precise "ground-truth" target states
        # are computed with 10-fold larger number of Trotter layers.
        # Parameter "evol_times" provides physical timestamps per horizon.
        self.trotter_steps = step_range * int(round(big_step / small_step))
        self.evol_times = np.round(step_range * big_step, 3)

        # As we go forward in time, the ansatz circuit will be expanded by
        # this number of layers on every big step.
        self.num_layers_inc = int(2)
        # Alternatively, if user defined his/her own schedule, the number
        # of ansatz layers for a time horizon will be picked up from the list.
        # In case of None or a short list, we fall back to the constant increment.
        self.manual_num_layers = None  # [2, 4, 6, 7, 8, 9, 10, 11]

        # Type of objective function: "sur_max" or "sur_fast_mps_trotter".
        # Either the full vectors or the ones in MPS format are used.
        # The former approach is limited to a moderate number of qubits and
        # used as a reference. The latter one relies on MPS representation
        # and enables simulation with large number of qubits.
        self.objective = "sur_fast_mps_trotter"
        # self.objective = "sur_max"

        # Function that creates a quantum circuit for initial state preparation.
        # Signature: my_init_circuit(num_qubits: int) -> QuantumCircuit
        # Examples: identity_circuit(), half_zero_circuit(), alt_init_circuit()
        # or a custom one complying with the signature.
        self.ini_state_func = (trotop.neel_init_state,)

        # Maximum number of optimization iterations.
        self.maxiter = int(40)

        # Time limit for optimization in seconds; -1 means no limit.
        self.time_limit = int(-1)

        # Seed to initialize the pseudo-random generator. Hard-coded value
        # allows reproducibility of the previously obtained results.
        self.seed = int(round(time.time()))

        # Desired least fidelity. None implies automatic selection.
        self.fidelity_thr = float(0.995)

        # Enables 2nd order Trotter circuit, recommended.
        self.second_order_trotter = True

        # Enables verbosity.
        self.verbose = True

        # Experimental. Enables gradient scaling, i.e. amplification of
        # vanishing gradient on barren plateau to prevent early termination.
        self.enable_grad_scaling = True

        # Debugging. Enables storing the intermediate optimization results.
        self.save_intermediate_results = False

    @property
    def use_mps(self) -> bool:
        """Use of MPS or full vectors depending on objective function."""
        return bool(self.objective.find("mps") >= 0)
