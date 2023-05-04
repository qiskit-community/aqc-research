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
Parametric circuit classes. We often call a parametric circuit an ansatz or
a parametric ansatz.
"""

from abc import ABC
from typing import Optional, Union, Tuple
import numpy as np
import aqc_research.checking as chk


class ParametricCircuit(ABC):
    """
    Class represents a parametric circuit. It can also serve as a base class
    for a custom parametrized circuit.

    A unit-block is assumed having the following structure (CZ and CPhase
    blocks are similar, with the same set of 1-qubit gates):

    control ---*---|Ry|-|Rz|--       ---*---|Ry|-|Rz|--       ---*---|Ry|-|Rz|--
               |                 or     |                 or     |
    target  --|CX|-|Ry|-|Rx|--       --|CZ|-|Ry|-|Rz|--       --|CP|-|Ry|-|Rz|--
    """

    def __init__(
        self,
        num_qubits: int,
        entangler: str,
        blocks: np.ndarray,
        name: Optional[str] = None,
        power: Optional[int] = 1,
    ):
        """
        **Note**: consider that "blocks" can be changed by some algorithms
        during optimization.

        Args:
            num_qubits: number of qubits.
            entangler: type of entangling gate, one of: ["cx", "cz", "cp"].
            blocks: an external circuit structure created by user; an array of
                    dimensions ``(2, depth)`` indicating where the 2-qubit
                    unit-blocks will be placed; the 1st row contains indices of
                    control inputs, while target indices occupy the 2nd row.
            name: optional circuit name.
            power: the circuit can be repeated ``power`` times, i.e. the
                   equivalent ansatz matrix should be ``V^power``;
                   **experimental**, must be equal to 1 for now.
        """
        self.check_block_layout(num_qubits, blocks)
        if not chk.is_int(power, power >= 1):
            raise ValueError("expects circuit power (V^p) to be integer and p >= 1")

        self._num_qubits = int(num_qubits)
        self._thetas_per_block = int(5 if entangler == "cp" else 4)
        self._blocks = blocks.astype(int).copy()
        self._entangler = entangler
        self._name = name if isinstance(name, str) else ""
        self._power = int(power)

    def update_structure(self, blocks: np.ndarray):
        """
        Initialize this parametric circuit with a new structure of unit-blocks.
        """
        self.check_block_layout(self.num_qubits, blocks)
        self._blocks = blocks.astype(int).copy()

    @property
    def name(self) -> str:
        """
        Returns circuit name, if specified.
        """
        return self._name

    @property
    def num_qubits(self) -> int:
        """
        Returns the number of qubits.
        """
        return int(self._num_qubits)

    @property
    def dimension(self) -> int:
        """
        Returns the problem dimensionality, dim = 2^(num qubits).
        """
        return int(2**self.num_qubits)

    @property
    def num_blocks(self) -> int:
        """
        Returns the number of unit-blocks in this parametric circuit.
        """
        return int(self._blocks.shape[1])

    @property
    def num_thetas(self) -> int:
        """
        Returns the number of angular parameters in this parametric circuit.
        """
        return int(3 * self.num_qubits + self.tpb * self.num_blocks)

    @property
    def blocks(self) -> np.ndarray:
        """
        Returns the structure of unit-blocks as an array of size (2, depth).
        """
        return self._blocks

    @property
    def entangler(self) -> str:
        """
        Returns the name of block's entangling gate, "cx", "cz" or "cp".
        """
        return self._entangler

    @property
    def tpb(self) -> int:
        """
        Returns the number of theta parameters per unit-block.
        """
        return int(self._thetas_per_block)

    @property
    def circuit_power(self) -> int:
        """
        Returns number of times the ansatz circuit is repeated, ``V^power``.
        **Note**: experimental, must be equal to 1 for now.
        """
        return int(self._power)

    def subset1q(
        self,
        vec: np.ndarray,  # vec: Union[np.ndarray, th.Tensor]
    ) -> np.ndarray:  # ) -> Union[np.ndarray, th.Tensor]:
        """
        Returns a subset of vector entries that correspond to the parameters of
        1-qubit front gates. Function is useful for manipulation with vectors
        of parameters and gradients when we need to address only 1-qubit gates.

        Args:
            vec: vector of parameters or gradients whose length is equal to
                 ``self.num_thetas``.

        Returns:
            a view of sub-vector of input vector, where the view covers
            the entries corresponding to 1-qubit front gates; it is reshaped
            to 2D array of size ``(number_of_qubits, 3)``.
        """
        # assert isinstance(vec, (np.ndarray, th.Tensor))
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (self.num_thetas,)
        return vec[0 : 3 * self.num_qubits].reshape(-1, 3)

    def subset2q(
        self,
        vec: np.ndarray,  # vec: Union[np.ndarray, th.Tensor]
    ) -> np.ndarray:  # ) -> Union[np.ndarray, th.Tensor]:
        """
        Returns a subset of vector entries that correspond to the parameters of
        2-qubit gates. Function is useful for manipulation with vectors of
        parameters and gradients when we need to address only 2-qubit gates.

        Args:
            vec: vector of parameters or gradients whose length is equal to
                 ``self.num_thetas``.

        Returns:
            a view of sub-vector of input vector, where the view covers the
            entries corresponding to 2-qubit gates; it is reshaped to 2D array
            of size ``(number_of_unit_blocks, number_of_thetas_per_block)``.
        """
        # assert isinstance(vec, (np.ndarray, th.Tensor))
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (self.num_thetas,)
        return vec[3 * self.num_qubits :].reshape(-1, self.tpb)

    def insert_unit_blocks(
        self,
        pos: int,
        extra_blocks: np.ndarray,
        thetas: Optional[np.ndarray] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        """
        Inserts unit-blocks at specified block position and updates the angular
        parameters accordingly.

        Args:
            pos: block position where to insert new blocks; if ``pos`` is equal
                 to number of existing blocks, the new ones will be appended.
            extra_blocks: blocks to be inserted; shape: ``(2, #new_blocks)``.
            thetas: optional angular parameters that will be expanded and
                    padded by zeros to fit the new circuit structure; the size
                    of ``thetas`` vector should be equal the current depth of
                    the circuit (before expansion).

        Returns:
            1. expanded angular parameters padded by zeros at the positions
               of newly added blocks; if ``thetas`` were not specified, None
               will be returned.
            2. subset of indices where newly inserted parameters have been
               placed, if ``thetas`` was specified, otherwise None; one can
               use these indices to initialize the new angular parameters
               to something different from zero.
        """
        self.check_block_layout(self.num_qubits, extra_blocks)
        assert chk.is_int(pos, 0 <= pos <= self.num_blocks)
        assert (thetas is None) or chk.float_1d(thetas)
        assert (thetas is None) or thetas.size == self.num_thetas

        new_idx = None
        self._blocks = np.insert(self._blocks, [pos], extra_blocks, axis=1)
        if thetas is not None:
            tpos = 3 * self.num_qubits + pos * self.tpb
            size = self.tpb * extra_blocks.shape[1]
            zero_arr = np.zeros(size, dtype=thetas.dtype)
            thetas = np.insert(thetas, [tpos], zero_arr)
            new_idx = np.arange(tpos, tpos + zero_arr.size, dtype=int)
            assert thetas.size == self.num_thetas

        return thetas, new_idx

    def check_block_layout(self, num_qubits: int, blocks: np.ndarray):
        """
        Checks if a valid, generic unit-block structure has been specified.

        Args:
            num_qubits: number of qubits.
            blocks: structure of unit-blocks that defines circuit layout.

        Raises:
            ValueError exception if not a valid structure was specified.
        """
        if not (
            chk.is_int(num_qubits)
            and num_qubits >= 2
            and isinstance(blocks, np.ndarray)
            and blocks.dtype in (int, np.int32, np.int64)
            and blocks.shape == (2, blocks.size // 2)
            and np.all(np.logical_and(0 <= blocks, blocks < num_qubits))
            and np.all(blocks[0, :] != blocks[1, :])
        ):
            raise ValueError("not a valid structure of unit-blocks")

    @property
    def num_layers(self) -> int:
        """Returns the number of layers. Not applicable to generic circuit."""
        raise NotImplementedError("there are no layers in generic ansatz")

    @property
    def bpl(self) -> int:
        """Returns the number blocks per layer. Not applicable to generic circuit."""
        raise NotImplementedError("there are no layers in generic ansatz")


class TrotterAnsatz(ParametricCircuit):
    """
    Parametric circuit that consists of triple-block Trotter structures combined
    into layers. Every triple block connects a couple of adjacent qubits and
    resembles an elementary Trotter circuit.

    There are ``(n-1)`` triplets in each layer. In case of second order Trotter,
    an extra half-layer with ``n/2`` triplets is added at he end.

    **Note**, if 2nd order Trotter is intended, a *core function is responsible*
    to add a half-layer at the end of the circuit. The latter half-layer is *not*
    explicitly reflected in this ansatz but implied. Recall, the core routines
    compute objective function and its gradient.

    **Note**, in order to simplify implementation we assume that 2nd order
    Trotter ansatz has one extra ("virtual") half-layer at the end, which takes
    exactly the same angular parameters as the very first half-layer. That is,
    the first and trail half-layers of 2-qubit unit-blocks are identical, and the
    total number of parameters is the same as in a 1st order Trotter ansatz with
    same number of full layers (as returned by the member function num_layers()).
    Technically this means that we do not increase the expressibility comparing
    to the 1st order ansatz (given the number of full layers), rather, we improve
    approximation accuracy by adding a trailing half-layer with exactly the same
    parameters as in the the leading half-layer.

    **Note**, there might be some confusion in case of 2nd order Trotter because
    the function ``num_blocks()`` returns the number of unit-blocks in the *full*
    layers ignoring the extra ("virtual") half-layer at the end. Also, one must
    add up the gradients by the angular parameters of the leading and trailing
    half-layers since these half-layers are supposed to be identical.

    """

    def __init__(
        self,
        num_qubits: int,
        blocks: np.ndarray,
        second_order: bool,
        name: Optional[str] = None,
    ):
        """
        Args:
            num_qubits: number of qubits.
            blocks: an external circuit structure created by user; an array of
                    dimensions ``(2, depth)`` indicating where the 2-qubit
                    unit-blocks will be placed; *important*, in case of 2nd
                    order Trotter, this structure should *not* include blocks
                    of the trailing half-layer, they are *implicitly* implied.
            second_order: True, if the 2nd order Trotter is intended.
            name: optional circuit name.
        """
        assert isinstance(second_order, bool)
        self._second_order = second_order  # before the base class constructor!
        super().__init__(num_qubits, "cx", blocks, name)

    @property
    def is_second_order(self) -> bool:
        """Returns True in case 2nd order Trotter."""
        return self._second_order

    @property
    def half_layer_num_blocks(self) -> int:
        """
        Returns the number of unit-blocks in leading/trailing half-layers
        in case of 2nd order Trotter, otherwise returns zero.
        """
        return int(3 * (self.num_qubits // 2)) if self._second_order else int(0)

    @property
    def num_layers(self) -> int:
        """
        Returns the number of **full** layers. Recall, 2nd order Trotter has
        additional trailing half-layer, identical to the front half-layer,
        which is **not** counted but implied.
        """
        return int(self.num_blocks // self.bpl)

    @property
    def bpl(self) -> int:
        """Returns the number of blocks per **full** layer."""
        return int(3 * (self.num_qubits - 1))

    def insert_unit_blocks(
        self,
        pos: int,
        extra_blocks: np.ndarray,
        thetas: Optional[np.ndarray] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        """
        Inserts unit-blocks at specified block position and updates the angular
        parameters accordingly. The number of extra blocks must be a multiple
        of the number of block per *full* layer, that is, one should insert a
        collection of full layers rather than an arbitrary set of unit-blocks.

        **Note**, although not explicitly stored, the blocks on the trailing
        half-layer are always implied for the 2nd order Trotter. That means
        the insertion operation is always applied to the chain of full layers,
        while the trailing half-layer is implicitly added at the end in any case.

        Args:
            pos: block position where to insert new blocks;
                 if ``block_pos`` is equal to number of existing blocks,
                 the new ones will be just appended.
            extra_blocks: blocks to be inserted; shape: ``(2, #new_blocks)``.
            thetas: optional angular parameters that will be extended and
                    padded by zeros to fit the new circuit structure; the size
                    of ``thetas`` vector should match the current depth of
                    the circuit (before it was extended).

        Returns:
            1. extended angular parameters padded by zeros at the positions
            of newly added blocks; if thetas were not specified, None will
            be returned.
            2. subset of indices where newly inserted parameters have been
            placed, if ``thetas`` was specified, otherwise None; one can
            use these indices to initialize the new angular parameters
            to something different from zero.
        """
        assert chk.is_int(pos, 0 <= pos <= self.num_blocks)
        self.check_block_layout(self.num_qubits, extra_blocks)
        if not bool(pos % (3 * (self.num_qubits - 1)) == 0):
            raise ValueError("position of blocks insertion must be aligned at layer boundary")
        return super().insert_unit_blocks(pos, extra_blocks, thetas)

    def check_block_layout(self, num_qubits: int, blocks: np.ndarray):
        """
        Checks if a valid, Trotter-like unit-block structure has been specified.
        Conditions: (1) expects layers of triplets; (2) 1st and 3rd blocks in
        a triplet are applied to the same qubits; (3) 2nd block in a triplet is
        flipped over; (4) 1st block is applied to successive, adjacent qubits.

        Args:
            num_qubits: number of qubits.
            blocks: structure of unit-blocks that defines circuit layout.

        Raises:
            ValueError exception if not a valid structure was specified.
        """
        super().check_block_layout(num_qubits, blocks)
        num_blocks = blocks.shape[1]
        if num_blocks > 0:
            bls = blocks.reshape((2, -1, 3))  # reshape into block triplets
            if not (
                num_blocks % (3 * (num_qubits - 1)) == 0
                and np.all(bls[:, :, 0] == bls[:, :, 2])  # 1st block == 3rd block
                and np.all(bls[0, :, 0] == bls[1, :, 1])  # flipped over 2nd block
                and np.all(bls[1, :, 0] == bls[0, :, 1])  # flipped over 2nd block
                and np.all(bls[0, :, 0] == bls[1, :, 0] + 1)  # adjacent qubits
            ):
                raise ValueError("not a valid Trotterized block layout")

            if self._second_order:
                # In the leading half-layer qubits connected by triplets are:
                # 0-1, 2-3, 4-5, etc.
                for i in range(num_qubits // 2):
                    if not (bls[0, i, 1] == 2 * i and bls[1, i, 1] == 2 * i + 1):
                        raise ValueError("unexpected layout of the leading half-layer")


def layer_to_block_range(
    circ: ParametricCircuit, layer_range: Union[Tuple[int, int], None]
) -> Tuple[int, int]:
    """
    Converts a range of layers into the range of corresponding unit-blocks.

    **Note**, generic ParametricCircuit does *not* necessary have a layered
    structure. Therefore, the input object ``circ`` implies some kind of ansatz
    with a regular layout, for example, the ``TrotterAnsatz`` one. In particular,
    the property functions ``num_layers`` and ``bpl`` must be implemented.

    Args:
        circ: parametric circuit with layered structure.
        layer_range: range of layers to be converted or None.

    Returns:
        sub-range of unit-blocks or the full range, if ``layer_range`` is None.
    """
    assert isinstance(circ, ParametricCircuit)
    if layer_range is None:
        return 0, circ.num_blocks
    assert chk.is_tuple(layer_range, len(layer_range) == 2)
    assert 0 <= layer_range[0] < layer_range[1] <= circ.num_layers
    block_range = (layer_range[0] * circ.bpl, layer_range[1] * circ.bpl)
    assert 0 <= block_range[0] < block_range[1] <= circ.num_blocks
    return block_range


def first_layer_included(
    circ: ParametricCircuit, layer_range: Union[Tuple[int, int], None]
) -> bool:
    """
    Returns True, if the first layer is included in the layer range or
    ``layer_range`` is None, implying the full range.
    """
    assert isinstance(circ, ParametricCircuit)
    if layer_range is None:
        return True
    assert chk.is_tuple(layer_range, len(layer_range) == 2)
    assert 0 <= layer_range[0] < layer_range[1] <= circ.num_layers
    return layer_range[0] == 0
