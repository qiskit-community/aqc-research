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
Tests the module 'core_op_matrix.py'.
"""

from typing import List, Tuple
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RZGate, RYGate
from qiskit.test import QiskitTestCase
import qiskit.quantum_info as qinfo
import aqc_research.utils as helper
from aqc_research.parametric_circuit import ParametricCircuit
import test.utils_for_testing as tut
import aqc_research.core_op_matrix as cop
import aqc_research.circuit_transform as ctr
import aqc_research.checking as chk


class TestCoreOperationsLevel2(QiskitTestCase):
    """Tests the module 'core_op_matrix.py'."""

    num_repeats = 2  # number of test repetitions
    max_num_qubits = 5  # max. number of qubits
    tol = float(np.sqrt(np.finfo(float).eps))  # tolerance

    def setUp(self):
        super().setUp()
        np.random.seed(0x0696969)

    @staticmethod
    def _gate_1q_mat(angles: np.ndarray) -> np.ndarray:
        """Computes a universal 2x2 1-qubit gate matrix from three angles."""
        assert angles.size == 3
        rz0 = RZGate(angles[0]).to_matrix()
        ry1 = RYGate(angles[1]).to_matrix()
        rz2 = RZGate(angles[2]).to_matrix()
        return rz2 @ ry1 @ rz0

    @staticmethod
    def _block_mat(
        *,
        num_qubits: int,
        ctrl: int,
        targ: int,
        c_angles: np.ndarray,
        t_angles: np.ndarray,
        g_angle: float,
        entangler: str,
    ) -> np.ndarray:
        """
        Creates matrix of 2-qubit block using Qiskit.
                                _______
            control -----*-----| c_mat |-----
                         |      -------
                         |      _______
            target  ----|G|----| t_mat |-----
                                -------

        Args:
            num_qubits: number of qubits.
            ctrl: index of control qubit.
            targ: index of target qubit.
            c_angles: angles that make up control gate (c_mat).
            t_angles: angles that make up target gate (t_mat).
            g_angle: angles that make up entangling gate (g_mat), if applicable.
            entangler: entangling gate, {"cx", "cz", "cp"}.

        Returns:
            unit-block matrix expanded to the full size.
        """
        assert c_angles.shape == t_angles.shape == (3,)
        assert entangler in {"cx", "cz", "cp"}
        qcirc = QuantumCircuit(num_qubits)
        if entangler == "cp":
            qcirc.cp(g_angle, ctrl, targ)
        elif entangler == "cz":
            qcirc.cz(ctrl, targ)
        else:
            qcirc.cx(ctrl, targ)
        qcirc.rz(c_angles[0], ctrl)
        qcirc.ry(c_angles[1], ctrl)
        qcirc.rz(c_angles[2], ctrl)
        qcirc.rz(t_angles[0], targ)
        qcirc.ry(t_angles[1], targ)
        qcirc.rz(t_angles[2], targ)
        return Operator(qcirc).data

    def _num_qubits_range(self) -> list:
        """Returns a range of qubit numbers repeated several times."""
        return [n for n in range(2, self.max_num_qubits + 1) for _ in range(self.num_repeats)]

    @staticmethod
    def _all_controls_and_targets(num_qubits: int) -> List[Tuple]:
        """
        Returns a list of all admissible control/target qubits combinations
        given the number of qubits.
        """
        return [(c, t) for c in range(num_qubits) for t in range(num_qubits) if c != t]

    @staticmethod
    def _exact_gradient(
        circ: ParametricCircuit,
        thetas: np.ndarray,
        x_mat: np.ndarray,
        y_mat: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the exact gradient of the dot product <V@x|y> using
        parameter shift method. Straightforward and slow routine.
        """
        assert chk.complex_2d(x_mat) and chk.complex_2d(y_mat)
        assert x_mat.shape == y_mat.shape and x_mat.shape[0] >= x_mat.shape[1]
        cphase = np.zeros(thetas.size, dtype=bool)
        if circ.entangler == "cp":
            circ.subset2q(cphase)[:, 4] = True  # at positions of CPhase parameter
        th_tau = thetas.copy()
        grad = np.zeros(thetas.size, dtype=np.cfloat)
        for i in range(thetas.size):
            tau, scale = (np.pi / 2, 0.5) if cphase[i] else (np.pi, 0.25)
            th_tau[i] = thetas[i] - tau
            v_m = ctr.ansatz_to_numpy_by_qiskit(circ, th_tau)
            th_tau[i] = thetas[i] + tau
            v_p = ctr.ansatz_to_numpy_by_qiskit(circ, th_tau)
            grad[i] = scale * np.vdot((v_p - v_m) @ x_mat, y_mat)
            th_tau[i] = thetas[i]
        return grad

    def test_rotation_mul_mat(self):
        """
        Tests correctness of [rotation_gate]_mul_mat() functions.
        """
        tol = self.tol
        for num_qubits in self._num_qubits_range():
            workspace = np.zeros(4**num_qubits, dtype=np.cfloat)
            for qubit_no in range(num_qubits):
                mat = tut.rand_rect_mat(2**num_qubits, np.random.randint(1, 2**num_qubits))
                angle = 2 * np.pi * np.random.rand()
                for pau in "xyz":  # Pauli matrices
                    qcirc = QuantumCircuit(num_qubits)
                    getattr(qcirc, f"r{pau}")(angle, qubit=qubit_no)  # qcirc.rp()
                    prod1 = qinfo.Operator(qcirc).data @ mat
                    prod2 = getattr(cop, f"r{pau}_mul_mat")(  # rp_mul_mat()
                        angle, qubit_no, mat.copy(), workspace
                    )
                    self.assertTrue(np.allclose(prod1, prod2, tol, tol))

    def test_control_mul_mat(self):
        """
        Tests correctness of [control_gate]_mul_mat() functions.
        """
        tol = self.tol
        for num_qubits in self._num_qubits_range():
            workspace = np.zeros(4**num_qubits, dtype=np.cfloat)
            for ctrl, targ in self._all_controls_and_targets(num_qubits):
                mat = tut.rand_rect_mat(2**num_qubits, np.random.randint(1, 2**num_qubits))
                angle = 2 * np.pi * np.random.rand()
                for gate in "xzp":  # CX, CZ or CPhase
                    qcirc = QuantumCircuit(num_qubits)
                    if gate == "p":
                        getattr(qcirc, f"c{gate}")(angle, ctrl, targ)  # qcirc.cp()
                    else:
                        getattr(qcirc, f"c{gate}")(ctrl, targ)  # qcirc.cx(), qcirc.cz()
                    prod1 = qinfo.Operator(qcirc).data @ mat
                    prod2 = getattr(cop, f"c{gate}_mul_mat")(
                        ctrl, targ, angle, mat.copy(), workspace
                    )
                    self.assertTrue(np.allclose(prod1, prod2, tol, tol))

    def test_pauli_dot_mat(self):
        """
        Tests correctness of functions for computation of: 0.5j * <Pauli@w|z>.
        """
        tol = self.tol
        for num_qubits in self._num_qubits_range():
            workspace = np.zeros(4**num_qubits, dtype=np.cfloat)
            for qubit_no in range(num_qubits):
                w_mat = tut.rand_rect_mat(2**num_qubits, np.random.randint(1, 2**num_qubits))
                z_mat = tut.rand_rect_mat(w_mat.shape[0], w_mat.shape[1])
                for pau in "xyz":  # Pauli matrices
                    qcirc = QuantumCircuit(num_qubits)
                    getattr(qcirc, pau)(qubit=qubit_no)  # qcirc.p()
                    prod1 = 0.5j * np.vdot(qinfo.Operator(qcirc).data @ w_mat, z_mat)
                    prod2 = getattr(cop, f"{pau}_dot_mat")(  # p_dot_mat()
                        qubit_no, w_mat, z_mat, workspace
                    )
                    self.assertTrue(np.allclose(prod1, prod2, tol, tol))

    def test_block_mat(self):
        """
        Tests correctness of the function ``block_mat()`` against
        straightforward Qiskit implementation.
        """
        tol = self.tol
        for num_qubits in self._num_qubits_range():
            workspace = np.zeros(4**num_qubits, dtype=np.cfloat)
            for entangler in ["cx", "cz", "cp"]:
                for ctrl, targ in self._all_controls_and_targets(num_qubits):
                    mat = tut.rand_rect_mat(2**num_qubits, np.random.randint(1, 2**num_qubits))
                    mat_id = id(mat)

                    # Generate random unit-block gates.
                    angles = helper.rand_thetas(7)
                    a_c, a_t, a_g = angles[0:3], angles[3:6], float(angles[6])
                    m_c = self._gate_1q_mat(a_c)
                    m_t = self._gate_1q_mat(a_t)

                    # Block times matrix using Qiskit.
                    b_mat = self._block_mat(
                        num_qubits=num_qubits,
                        ctrl=ctrl,
                        targ=targ,
                        c_angles=a_c,
                        t_angles=a_t,
                        g_angle=a_g,
                        entangler=entangler,
                    )
                    b_mat = b_mat @ mat

                    # Block times matrix using the core utilities.
                    ctrl_mul_mat = getattr(cop, f"{entangler}_mul_mat")
                    mat = ctrl_mul_mat(ctrl, targ, a_g, mat, workspace)
                    mat = cop.gate2x2_mul_mat(ctrl, m_c, mat, workspace)
                    mat = cop.gate2x2_mul_mat(targ, m_t, mat, workspace)

                    # Compare both implementations for equality.
                    self.assertTrue(id(mat) == mat_id)
                    self.assertTrue(np.allclose(b_mat, mat, tol, tol))

    def test_v_mul_mat(self):
        """
        Tests the functions ``v_mul_mat()`` and ``v_dagger_mul_mat()``.
        Expects: ``V @ V.H @ mat == V.H @ V @ mat == mat``.
        """
        tol = self.tol
        for num_qubits in self._num_qubits_range():
            workspace = np.zeros(4**num_qubits, dtype=np.cfloat)
            for entangler in ["cx", "cz", "cp"]:
                mat = tut.rand_rect_mat(2**num_qubits, np.random.randint(1, 2**num_qubits))

                # Generate a random circuit and theta parameters.
                circ = ParametricCircuit(
                    num_qubits=num_qubits,
                    entangler=entangler,
                    blocks=tut.rand_circuit(num_qubits, np.random.randint(40, 50)),
                )
                thetas = helper.rand_thetas(circ.num_thetas)

                # m2 = V @ V.H @ mat == mat.
                m_1 = cop.v_dagger_mul_mat(circ, thetas, mat.copy(), workspace)
                m1_id = id(m_1)
                m_2 = cop.v_mul_mat(circ, thetas, m_1, workspace)
                self.assertTrue(id(m_2) == m1_id)
                self.assertTrue(np.allclose(mat, m_2, atol=tol, rtol=tol))

                # m2 = V.H @ V @ mat == mat.
                m_1 = cop.v_mul_mat(circ, thetas, mat.copy(), workspace)
                m1_id = id(m_1)
                m_2 = cop.v_dagger_mul_mat(circ, thetas, m_1, workspace)
                self.assertTrue(id(m_2) == m1_id)
                self.assertTrue(np.allclose(mat, m_2, atol=tol, rtol=tol))

    def test_circuit_matrix(self):
        """
        Tests various methods for matrix construction from a parametric circuit.
        """
        tol = self.tol
        for num_qubits in self._num_qubits_range():
            workspace = np.zeros(4**num_qubits, dtype=np.cfloat)
            for entangler in ["cx", "cz", "cp"]:
                # Generate a random circuit and theta parameters.
                circ = ParametricCircuit(
                    num_qubits=num_qubits,
                    entangler=entangler,
                    blocks=tut.rand_circuit(num_qubits, np.random.randint(40, 50)),
                )
                thetas = helper.rand_thetas(circ.num_thetas)

                # Compute circuit matrices by three independent routines.
                qmat = ctr.ansatz_to_numpy_by_qiskit(circ, thetas)
                nmat = ctr.ansatz_to_numpy_trotter(circ, thetas)
                fmat = ctr.ansatz_to_numpy_fast(circ, thetas)
                vmat = cop.v_mul_mat(
                    circ, thetas, np.eye(2**num_qubits, dtype=np.cfloat), workspace
                )

                # Compare all matrices for equality.
                self.assertTrue(np.allclose(qmat, vmat, atol=tol, rtol=tol))
                self.assertTrue(np.allclose(qmat, nmat, atol=tol, rtol=tol))
                self.assertTrue(np.allclose(qmat, fmat, atol=tol, rtol=tol))

    def test_gradient(self):
        """
        Compares exact gradient, obtained by parameter shift method, against
        the fast gradient.
        """
        tol = self.tol
        for num_qubits in self._num_qubits_range():
            workspace = np.zeros(4**num_qubits, dtype=np.cfloat)
            for entangler in ["cx", "cz", "cp"]:
                print(".", end="", flush=True)  # print progress
                x_mat = tut.rand_rect_mat(2**num_qubits, np.random.randint(1, 2**num_qubits))
                y_mat = tut.rand_rect_mat(x_mat.shape[0], x_mat.shape[1])

                # Generate a random circuit and theta parameters.
                circ = ParametricCircuit(
                    num_qubits=num_qubits,
                    entangler=entangler,
                    blocks=tut.rand_circuit(num_qubits, np.random.randint(5, 10)),
                )
                thetas = helper.rand_thetas(circ.num_thetas)

                # Compute and compare both gradients for equality.
                exact_grad = self._exact_gradient(circ, thetas, x_mat, y_mat)
                fast_grad = cop.grad_of_matrix_dot_product(
                    circ=circ,
                    thetas=thetas,
                    x_mat=x_mat,
                    vh_y_mat=cop.v_dagger_mul_mat(circ, thetas, y_mat, workspace),
                    workspace=workspace,
                )
                self.assertTrue(np.allclose(exact_grad, fast_grad, tol, tol))
        print()

    def test_ansatz_to_numpy(self):
        """
        Tests conversion from parametric circuit to Numpy matrix.
        """
        tol = self.tol
        for num_qubits in self._num_qubits_range():
            for entangler in ["cx", "cz", "cp"]:
                circ = ParametricCircuit(  # random circuit
                    num_qubits=num_qubits,
                    entangler=entangler,
                    blocks=tut.rand_circuit(num_qubits, np.random.randint(10, 21)),
                )
                thetas = helper.rand_thetas(circ.num_thetas)
                mat1 = ctr.ansatz_to_numpy_by_qiskit(circ, thetas)
                mat2 = ctr.ansatz_to_numpy_trotter(circ, thetas)
                mat3 = ctr.ansatz_to_numpy_fast(circ, thetas)
                self.assertTrue(np.allclose(mat1, mat2, tol, tol))
                self.assertTrue(np.allclose(mat1, mat3, tol, tol))


if __name__ == "__main__":
    unittest.main()
