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
Tests for the module 'core_operations.py'.
**Note**, in Numpy implementation the bit ordering is made conformal to
Qiskit convention through the function ``bit2bit_transform()``.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import RZGate, RYGate
from qiskit.test import QiskitTestCase
import aqc_research.utils as helper
from aqc_research.parametric_circuit import ParametricCircuit, TrotterAnsatz
import test.utils_for_testing as tut
import aqc_research.core_operations as cop
import aqc_research.circuit_transform as ctr
from aqc_research.circuit_structures import make_trotter_like_circuit


class TestCoreOperation(QiskitTestCase):
    """Tests for the module 'core_operations.py'."""

    num_repeats = 2  # number of test repetitions
    max_num_qubits = 5  # max. number of qubits
    max_depth = 10  # max. circuit depth
    tol = float(np.sqrt(np.finfo(float).eps))  # tolerance

    def setUp(self):
        super().setUp()
        np.random.seed(0x0696969)

    @staticmethod
    def _full_1q_mat(num_qubits: int, pos: int, angles: np.ndarray) -> np.ndarray:
        """
        Computes full 1-qubit gate matrix using Qiskit.
        """
        qcirc = QuantumCircuit(num_qubits)
        qcirc.rz(angles[0], pos)
        qcirc.ry(angles[1], pos)
        return Operator(qcirc).data

    @staticmethod
    def _gate_1q_mat(angles: np.ndarray) -> np.ndarray:
        """
        Computes 2x2 1-qubit gate matrix from angles. The matrix is equivalent
        to the one generated by the function ``self._full_1q_mat()`` but it is
        not expanded to the full size.
        """
        assert angles.size == 2
        rz_ = RZGate(angles[0]).to_matrix()
        ry_ = RYGate(angles[1]).to_matrix()
        return ry_ @ rz_

    @staticmethod
    def _kron_1q_mat(num_qubits: int, pos: int, gate2x2: np.ndarray) -> np.ndarray:
        """
        Computes full 1-qubit gate matrix directly from Kronecker product.
        """
        return tut.kron3(np.eye(2**pos), gate2x2, np.eye(2 ** (num_qubits - pos - 1)))

    @staticmethod
    def _block_2q_mat(
        *,
        num_qubits: int,  # number of qubits
        ctrl: int,  # index of control qubit
        targ: int,  # index of target qubit
        c_angles: np.ndarray,  # angles that make up control gate (c_mat)
        t_angles: np.ndarray,  # angles that make up target gate (t_mat)
        g_angles: np.ndarray,  # angles that make up entangling gate (g_mat)
    ) -> np.ndarray:
        """
        Creates matrix of 2-qubit block using Qiskit.
                                _______
            control -----*-----| c_mat |-----
                         |      -------
                         |      _______
            target  ----|G|----| t_mat |-----
                                -------
        """
        assert c_angles.shape == t_angles.shape == g_angles.shape == (2,)
        ent = QuantumCircuit(1)  # sub-circuit of entangling gate
        ent.rz(g_angles[0], 0)
        ent.ry(g_angles[1], 0)
        custom = ent.to_gate().control(1)
        qcirc = QuantumCircuit(num_qubits)
        qcirc.append(custom, [ctrl, targ])
        qcirc.rz(c_angles[0], ctrl)
        qcirc.ry(c_angles[1], ctrl)
        qcirc.rz(t_angles[0], targ)
        qcirc.ry(t_angles[1], targ)
        return Operator(qcirc).data

    @staticmethod
    def _rand_circuit(num_qubits: int, entangler: str, second_order: bool) -> ParametricCircuit:
        """Generate a random circuit."""
        if entangler == "cx" and np.random.rand() >= 0.5:
            return TrotterAnsatz(
                num_qubits=num_qubits,
                blocks=make_trotter_like_circuit(
                    num_qubits=num_qubits, num_layers=np.random.randint(1, 7)
                ),
                second_order=second_order,
            )

        return ParametricCircuit(
            num_qubits=num_qubits,
            entangler=entangler,
            blocks=tut.rand_circuit(num_qubits=num_qubits, depth=np.random.randint(40, 50)),
        )

    def test_gate2x2_mul_vec(self):
        """
        Tests correctness of gate2x2_mul_vec() function.
        """
        b2b = cop.bit2bit_transform  # shorthand alias
        tol = self.tol
        for num_qubits in range(2, self.max_num_qubits + 1):
            for qubit in range(num_qubits):
                # Generic case.
                for _ in range(self.num_repeats):
                    # Straightforward computation of matrix-vector products.
                    phi = helper.rand_thetas(2)
                    vec = tut.rand_vec(2**num_qubits)
                    out = np.zeros_like(vec)
                    gate2x2 = self._gate_1q_mat(phi)

                    # Obtain the same result by Qiskit, Kron and core utility.
                    qs_prod = self._full_1q_mat(num_qubits, qubit, phi) @ vec

                    kr_prod = self._kron_1q_mat(num_qubits, b2b(num_qubits, qubit), gate2x2) @ vec

                    co_prod = cop.gate2x2_mul_vec(
                        num_qubits, b2b(num_qubits, qubit), gate2x2, vec, out, False
                    )

                    self.assertTrue(id(co_prod) == id(out))
                    self.assertTrue(np.allclose(qs_prod, co_prod, atol=tol, rtol=tol))
                    self.assertTrue(np.allclose(kr_prod, co_prod, atol=tol, rtol=tol))

                    # Compute in-place. Note, "vec" will be overwritten.
                    co_prod = cop.gate2x2_mul_vec(
                        num_qubits, b2b(num_qubits, qubit), gate2x2, vec, out, True
                    )

                    self.assertTrue(id(co_prod) == id(vec))
                    self.assertTrue(np.allclose(qs_prod, co_prod, atol=tol, rtol=tol))
                    self.assertTrue(np.allclose(kr_prod, co_prod, atol=tol, rtol=tol))

                # Special cases.
                for case_no in range(8):
                    gate2x2 = tut.rand_mat(2, kind="complex")
                    if case_no == 0:
                        gate2x2.fill(0)
                    elif case_no == 1:
                        gate2x2 = np.asarray([[1, 0], [0, 0]], dtype=np.cfloat)
                    elif case_no == 2:
                        gate2x2 = np.asarray([[0, 0], [0, 1]], dtype=np.cfloat)
                    elif case_no == 3:
                        gate2x2[0, :] = 0
                    elif case_no == 4:
                        gate2x2[1, :] = 0
                    elif case_no == 5:
                        gate2x2[:, 0] = 0
                    elif case_no == 6:
                        gate2x2[:, 1] = 0

                    vec = tut.rand_vec(2**num_qubits)
                    out = np.zeros_like(vec)

                    kr_prod = self._kron_1q_mat(num_qubits, b2b(num_qubits, qubit), gate2x2) @ vec
                    co_prod = cop.gate2x2_mul_vec(
                        num_qubits, b2b(num_qubits, qubit), gate2x2, vec, out, False
                    )

                    self.assertTrue(id(co_prod) == id(out))
                    self.assertTrue(np.allclose(kr_prod, co_prod, atol=tol, rtol=tol))

                    co_prod = cop.gate2x2_mul_vec(
                        num_qubits, b2b(num_qubits, qubit), gate2x2, vec, out, True
                    )

                    self.assertTrue(id(co_prod) == id(vec))
                    self.assertTrue(np.allclose(kr_prod, co_prod, atol=tol, rtol=tol))

    def test_block_mul_vec(self):
        """
        Tests correctness of the function ``block_mul_vec()`` against
        straightforward Qiskit implementation.
        """
        b2b = cop.bit2bit_transform  # shorthand alias
        tol = self.tol
        for num_qubits in range(2, self.max_num_qubits + 1):
            dim = 2**num_qubits
            workspace = np.zeros((2, dim), dtype=np.cfloat)

            for ctrl in range(num_qubits):  # for all control qubits ...
                for targ in range(num_qubits):  # for all target qubits ...
                    if ctrl == targ:
                        continue
                    for __ in range(self.num_repeats):
                        # Generate random 1-qubit gates and full-size vector.
                        angles = helper.rand_thetas(6)
                        a_c, a_t, a_g = angles[0:2], angles[2:4], angles[4:6]
                        m_c = self._gate_1q_mat(a_c)
                        m_t = self._gate_1q_mat(a_t)
                        m_g = self._gate_1q_mat(a_g)
                        vec = tut.rand_vec(dim)

                        # Block times vector using Qiskit.
                        qs_prod = (
                            self._block_2q_mat(
                                num_qubits=num_qubits,
                                ctrl=ctrl,
                                targ=targ,
                                c_angles=a_c,
                                t_angles=a_t,
                                g_angles=a_g,
                            )
                            @ vec
                        )

                        # Block times vector using the core utility.
                        co_prod = cop.block_mul_vec(
                            num_qubits,
                            b2b(num_qubits, ctrl),
                            b2b(num_qubits, targ),
                            m_c,
                            m_t,
                            m_g,
                            vec,
                            workspace,
                            dagger=False,
                        )

                        # Compare both implementations for equality.
                        self.assertTrue(id(co_prod) == id(vec))
                        self.assertTrue(np.allclose(qs_prod, co_prod, atol=tol, rtol=tol))

    def test_v_mul_vec(self):
        """
        Tests the functions ``v_mul_vec(..)`` and ``v_dagger_mul_vec(..)``.
        Expects: ``V @ V.H @ vec = V.H @ V @ vec``.
        """
        tol = self.tol
        for num_qubits in range(2, self.max_num_qubits + 1):
            dim = 2**num_qubits
            workspace = np.zeros((2, dim), dtype=np.cfloat)

            for second_order in [False, True]:
                for entangler in ["cx", "cz", "cp"]:
                    for repeat_no in range(self.num_repeats):
                        # Generate a random circuit and full-size vector.
                        circ = self._rand_circuit(num_qubits, entangler, second_order)
                        thetas = helper.rand_thetas(circ.num_thetas)
                        vec = tut.rand_vec(dim)
                        vec_orig = vec.copy()
                        out = vec if repeat_no % 2 == 0 else np.zeros_like(vec)

                        # vec2 = V @ V.H @ vec == vec.
                        vec1 = cop.v_dagger_mul_vec(circ, thetas, vec, out, workspace)
                        vec2 = cop.v_mul_vec(circ, thetas, vec1, out, workspace)
                        self.assertTrue(np.allclose(vec_orig, vec2, atol=tol, rtol=tol))

                        # vec2 = V.H @ V @ vec == vec.
                        np.copyto(vec, vec_orig)
                        vec1 = cop.v_mul_vec(circ, thetas, vec, out, workspace)
                        vec2 = cop.v_dagger_mul_vec(circ, thetas, vec1, out, workspace)
                        self.assertTrue(np.allclose(vec_orig, vec2, atol=tol, rtol=tol))

    def test_circuit_matrix(self):
        """
        Tests various methods for matrix construction from a parametric circuit.
        """
        tol = self.tol
        for num_qubits in range(2, self.max_num_qubits + 1):
            dim = 2**num_qubits
            workspace = np.zeros((2, dim), dtype=np.cfloat)

            for second_order in [False, True]:
                for entangler in ["cx", "cx", "cz", "cp"]:  # repeat "cx" for Trotter
                    for __ in range(self.num_repeats):
                        circ = self._rand_circuit(num_qubits, entangler, second_order)
                        thetas = helper.rand_thetas(circ.num_thetas)

                        # Convert to Qiskit circuit, then to numpy matrix.
                        qmat = ctr.ansatz_to_numpy_by_qiskit(circ, thetas)

                        # Convert to numpy matrix directly.
                        nmat = ctr.ansatz_to_numpy_trotter(circ, thetas)

                        # ansatz_to_numpy_fast() doesn't support Trotterized ansatz.
                        if isinstance(circ, TrotterAnsatz):
                            fmat = nmat
                        else:
                            fmat = ctr.ansatz_to_numpy_fast(circ, thetas)

                        # Create the circuit matrix by multiplying by every column
                        # (here, row which is the same as column) of the identity
                        # one, i.e. V @ vector_row[i], for all i.
                        vmat = np.eye(dim, dtype=np.cfloat)
                        for i in range(dim):
                            cop.v_mul_vec(circ, thetas, vmat[i], vmat[i], workspace)
                        vmat = vmat.T  # because we used rows instead of columns

                        # Compare all matrices for equality.
                        self.assertTrue(np.allclose(qmat, vmat, atol=tol, rtol=tol))
                        self.assertTrue(np.allclose(qmat, nmat, atol=tol, rtol=tol))
                        self.assertTrue(np.allclose(qmat, fmat, atol=tol, rtol=tol))


if __name__ == "__main__":
    unittest.main()
