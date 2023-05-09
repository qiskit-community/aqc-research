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
Utilities for creating a structure of parametrized ansatz circuit.
1) We call "unit block" a 2-qubit operator that comprises a single
entangling gate and four 1-qubit gates.
2) Depth of a circuit is defined as a total number of unit blocks.
3) Here, we do NOT create a circuit per se, rather a structure that defines
positions of unit blocks in the circuit.
"""

from typing import List, Optional
from logging import Logger
import numpy as np
from aqc_research.utils import create_logger
import aqc_research.checking as chk

_logger = create_logger(__file__)


def lower_limit(num_qubits: int) -> int:
    """
    Returns lower limit on the number of unit blocks that guarantees exact
    representation of a unitary operator by quantum gates.

    Args:
        num_qubits: number of qubits.

    Returns:
        lower limit on the number of unit blocks.
    """
    num_blocks = int(round(np.ceil((4**num_qubits - 3 * num_qubits - 1) / 4.0)))
    return num_blocks


def create_ansatz_structure(
    num_qubits: int,
    layout: str = "spin",
    connectivity: str = "full",
    depth: int = 0,
    block_repeat: int = 1,
    logger: Optional[Logger] = None,
) -> np.ndarray:
    """
    Generates a structure of parametrized circuit (ansatz) assuming the circuit
    consists of 2-qubit unit blocks each comprising an entangling gate and few
    single-qubit ones, for details see the papers:
    https://arxiv.org/abs/2106.05649 , https://arxiv.org/abs/2205.04025

    The inputs of 2-qubit unit block can be distinguished as "control" input
    and "target" one. The latter is particularly important for the unit blocks
    built from a single CNot gate and four 1-qubit rotations. An array returned
    by this function contains positions of "control" inputs in the first row,
    and corresponding "target" positions in the second one.

    Args:
        num_qubits: number of qubits.
        layout: type of circuit geometry, {"spin", "line", "cyclic_spin", "cyclic_line"}.
        connectivity: type of inter-qubit connectivity, {"full", "line"}.
        depth: circuit depth equals the total number of 2-qubit unit blocks;
               if non-positive, the maximum value will be chosen, which
               can be exponentially (!) large.
        block_repeat: unit blocks can repeat 1, 2 or 3 times one after another
                      while being connected to the same couple of qubits.
        logger: optional logging object for verbosity.

    Returns:
        A matrix of size ``(2, depth)`` which defines the placements of 2-qubit
        unit blocks in a parametric circuit, that is, the first row gives
        positions where the control inputs of unit blocks are connected,
        while the second row gives corresponding positions of the target inputs.

    Raises:
        ValueError: if unsupported type of circuit layout or number of qubits
        or combination of parameters is passed.
    """
    if num_qubits < 2:
        raise ValueError("Number of qubits must be greater or equal to 2")

    if depth <= 0 and layout != "front_layer_only":
        new_depth = lower_limit(num_qubits)
        if logger:
            logger.warning(f"choosing the maximum number of 2-qubit unit blocks: {new_depth}")
        depth = new_depth

    if not bool(1 <= block_repeat <= 3):
        raise ValueError("'block_repeat' argument must be equal 1, 2 or 3")

    if layout == "spin":
        _expect_line_or_full(layout=layout, conn=connectivity)
        blocks = _spin(num_qubits=num_qubits, depth=depth)

    elif layout == "line":
        _expect_line_or_full(layout=layout, conn=connectivity)
        blocks = _line(num_qubits, depth)

    elif layout == "cyclic_spin":
        _expect_line_or_full(layout=layout, conn=connectivity)
        blocks = _cyclic_spin(num_qubits, depth)

    elif layout == "cyclic_line":
        _expect_line_or_full(layout=layout, conn=connectivity)
        blocks = _cyclic_line(num_qubits, depth)

    else:
        raise ValueError(
            f"Unknown type of circuit layout, "
            f"expects one of {circuit_layout_list()}, got {layout}"
        )

    # Repeat the blocks, if requested.
    if block_repeat > 1:
        blocks = np.repeat(blocks, block_repeat, axis=1)

    if logger:
        logger.info(
            f"ansatz: connectivity='{connectivity}', layout='{layout}', "
            f"depth={depth}, unit-blocks repeat {block_repeat} times"
        )
    return blocks


def make_trotter_like_circuit(
    num_qubits: int,
    num_layers: int,
    *,
    connectivity: str = "full",
    verbose: bool = False,
) -> np.ndarray:
    """
    Generates a circuit that resembles Trotter one.

    Args:
        num_qubits: number of qubits.
        connectivity: type of inter-qubit connectivity, ``{"full", "line"}``.
        num_layers: number of layers, where each layer consists of
                    ``(num_qubits-1)`` triple-block structures.
        verbose: enables verbosity.

    Returns:
        A matrix of size ``(2, N)`` matrix that defines layers in circuit,
        where ``N`` is the number of unit-blocks.

    Raises:
        ValueError: if unsupported type of circuit layout or number of qubits
        or combination of parameters are passed.
    """
    if num_qubits < 2:
        raise ValueError("number of qubits must be greater or equal to 2")
    if connectivity not in ["full", "line"]:
        raise ValueError("expects 'full' or 'line' connectivity")
    if num_layers < 0:
        raise ValueError("expects non-negative number of layers")
    if num_layers == 0:
        return np.zeros((2, 0), dtype=int)
    if verbose:
        _logger.info("Makes Trotter-like block structure with %d layers", num_layers)

    # Unit-blocks are repeated 3 times (triplets of blocks).
    blocks = _spin(num_qubits=num_qubits, depth=num_layers * (num_qubits - 1))
    blocks = np.repeat(blocks, 3, axis=1)
    # Swap control and target qubits in 1st and 3rd blocks of every triplet.
    bls = blocks.reshape((2, -1, 3))
    tmp = bls.copy()
    bls[0, :, [0, 2]] = tmp[1, :, [0, 2]]
    bls[1, :, [0, 2]] = tmp[0, :, [0, 2]]
    blocks = bls.reshape((2, -1)).copy()
    return blocks


def circuit_layout_list() -> List[str]:
    """
    Returns the list of available circuit layouts.
    """
    return [
        "spin",
        "line",
        "cyclic_spin",
        "cyclic_line",
    ]


def circuit_connectivity_list() -> List[str]:
    """
    Returns the list of available connectivity types.
    """
    return [
        "full",
        "line",
    ]


def num_blocks_per_layer(num_qubits: int, circuit_layout: str) -> int:
    """Returns the number of unit-blocks per layer."""
    assert chk.is_int(num_qubits, num_qubits >= 2)
    assert isinstance(circuit_layout, str) and circuit_layout in circuit_layout_list()
    return num_qubits if circuit_layout.startswith("cyclic_") else (num_qubits - 1)


def fraction_of_lower_bound(depth_fraction: float, num_qubits: int, circuit_layout: str) -> int:
    """
    Computes the number of layers in a circuit from the circuit length given
    as a fraction of the lower-bound depth, which guaranties exact compiling, see
    the paper https://arxiv.org/pdf/2106.05649.pdf.
    Returned value will be rounded to the nearest upper-bound integer.

    Default ansatz has a regular, layered structure with the following depth:
    ``circuit depth = (number of layers) * (number of blocks in a layer)``.

    The notion of layer is applicable to a regular circuit layout, e.g. "spin"
    structure. In the latter case, layer is formed by a subset of consecutive
    2-qubit unit-blocks where every pair of adjacent qubits is connected by a
    block. That is, there are ``num_qubits - 1`` unit-blocks in a layer.

    One can also choose "cyclic_spin" layout, additionally connecting the first
    and the last qubits by one extra block, with ``num_qubits`` blocks in a layer.

    **Note**, the maximal circuit depth can be a very large number of order
    ``O(4^n)``. That long circuit can be totally impractical and next to
    impossible to optimize in any reasonable time. Therefore, the fraction
    should be set to a quite small value depending on the number of qubits.

    Args:
        depth_fraction: a fraction of the maximal number of layers;
                        a floating point value between 0 and 1.
        num_qubits: number of qubits.
        circuit_layout: circuit layout; see the function ``circuit_layout_list()``.

    Returns:
        the number of layers in a circuit
        rounded to the nearest upper-bound integer.
    """
    assert chk.is_float(depth_fraction)
    if not bool(circuit_layout in circuit_layout_list()):
        raise ValueError(f"'circuit_layout' must be one of {circuit_layout_list()}")
    if not bool(0 < depth_fraction <= 1):
        raise ValueError("expects: 0 < depth_fraction <= 1")
    bpl = num_blocks_per_layer(num_qubits, circuit_layout)
    circuit_depth = int(round(depth_fraction * lower_limit(num_qubits)))
    num_layers = int(max(1, (circuit_depth + bpl - 1) // bpl))
    return num_layers


def _expect_line_or_full(layout: str, conn: str):
    """
    Checks that connectivity is either "line" or "full".
    """
    assert isinstance(layout, str) and isinstance(conn, str)
    if not (conn in ["line", "full"]):
        raise ValueError(f"layout '{layout}' assumes 'line' or 'full' connectivity, got {conn}")


def _spin(num_qubits: int, depth: int) -> np.ndarray:
    """
    Generates a spin-like circuit structure.

    Args:
        num_qubits: number of qubits.
        depth: depth of the circuit (number of unit blocks).

    Returns:
        a matrix of size ``2 x depth`` that defines positions of unit blocks.
    """
    layer = 0
    blocks = np.zeros((2, depth), dtype=int)
    while True:
        for shift in range(2):
            for i in range(shift, num_qubits - 1, 2):
                blocks[0, layer] = i
                blocks[1, layer] = i + 1
                layer += 1
                if layer >= depth:
                    return blocks


def _line(num_qubits: int, depth: int) -> np.ndarray:
    """
    Generates a line structure where the first and the last qubits
    are **not** connected. Circuit layout pattern repeats after every
    ``#qubits`` blocks.

    Args:
        num_qubits: number of qubits.
        depth: depth of the circuit (number of unit blocks).

    Returns:
        a matrix of size ``2 x depth`` that defines positions of unit blocks.
    """
    blocks = np.zeros((2, depth), dtype=int)
    pos = 0
    for i in range(depth):
        if pos % num_qubits == num_qubits - 1:
            pos += 1  # skip connecting the first and last qubits
        blocks[0, i] = (pos + 0) % num_qubits
        blocks[1, i] = (pos + 1) % num_qubits
        pos += 1
    return blocks


def _cyclic_spin(num_qubits: int, depth: int) -> np.ndarray:
    """
    Same as in the spin-like circuit but the first and the last qubits
    are also connected. Circuit layout pattern repeats after every
    ``#qubits`` blocks.

    Args:
        num_qubits: number of qubits.
        depth: depth of the circuit (number of unit blocks).

    Returns:
        a matrix of size ``2 x depth`` that defines positions of unit blocks.
    """
    blocks = np.zeros((2, depth), dtype=int)
    n_even = bool(num_qubits & 1 == 0)
    for i in range(depth):
        offset = (i // (num_qubits // 2)) % 2 if n_even else 0
        blocks[0, i] = (2 * i + offset + 0) % num_qubits
        blocks[1, i] = (2 * i + offset + 1) % num_qubits
    return blocks


def _cyclic_line(num_qubits: int, depth: int) -> np.ndarray:
    """
    Generates a line structure where the first and the last qubits
    are also connected. Circuit layout pattern repeats after every
    ``#qubits`` blocks.

    Args:
        num_qubits: number of qubits.
        depth: depth of the circuit (number of unit blocks).

    Returns:
        a matrix of size ``2 x depth`` that defines positions of unit blocks.
    """
    blocks = np.zeros((2, depth), dtype=int)
    for i in range(depth):
        blocks[0, i] = (i + 0) % num_qubits
        blocks[1, i] = (i + 1) % num_qubits
    return blocks
