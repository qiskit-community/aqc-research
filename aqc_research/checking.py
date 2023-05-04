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
Commonly used routines for checking the validity of variables and function arguments.
"""

import numpy as np


def is_str(x: str, condition: bool = True) -> bool:
    """Checks the variables is of type 'string'."""
    return isinstance(x, str) and condition


def is_int(x: int, condition: bool = True) -> bool:
    """Checks the variables is of type 'integer'."""
    return isinstance(x, (int, np.int32, np.int64)) and condition


def is_float(x: float, condition: bool = True) -> bool:
    """Checks the variables is of type 'float'."""
    return isinstance(x, (float, np.float32, np.float64)) and condition


def is_complex(x: np.cfloat, condition: bool = True) -> bool:
    """Checks the variables is of type 'complex float'."""
    return isinstance(x, (complex, np.complex64, np.complex128)) and condition


def is_bool(x: bool) -> bool:
    """Checks the variables is of type 'bool'."""
    return isinstance(x, bool)


def is_dict(x: dict, condition: bool = True) -> bool:
    """Checks the variables is of type 'dictionary'."""
    return isinstance(x, dict) and condition


def is_list(x: list, condition: bool = True) -> bool:
    """Checks the variables is of type 'list'."""
    return isinstance(x, list) and condition


def is_tuple(x: tuple, condition: bool = True) -> bool:
    """Checks the variables is of type 'tuple'."""
    return isinstance(x, tuple) and condition


def float_1d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '1D array of floats'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 1
        and x.dtype in (float, np.float32, np.float64)
        and condition
    )


def float_2d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '2D array of floats'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.dtype in (float, np.float32, np.float64)
        and condition
    )


def complex_array(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type 'an array of complex floats'."""
    return (
        isinstance(x, np.ndarray)
        and x.data.contiguous
        and x.dtype in (complex, np.complex64, np.complex128)
        and condition
    )


def complex_1d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '1D array of complex floats'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 1
        and x.dtype in (complex, np.complex64, np.complex128)
        and condition
    )


def complex_or_float_1d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '1D array real or complex floats'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 1
        and x.dtype in (float, np.float32, np.float64, complex, np.complex64, np.complex128)
        and condition
    )


def complex_2d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '2D array of complex floats'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.dtype in (complex, np.complex64, np.complex128)
        and condition
    )


def complex_3d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '3D array of complex floats'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 3
        and x.dtype in (complex, np.complex64, np.complex128)
        and condition
    )


def complex_or_float_2d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '2D array of real or complex floats'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.dtype in (float, np.float32, np.float64, complex, np.complex64, np.complex128)
        and condition
    )


def complex_2d_square(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '2D square array of complex floats'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.shape[0] == x.shape[1]
        and x.dtype in (np.complex64, np.complex128)
        and condition
    )


def int_1d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '1D array of integers'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 1
        and x.dtype in (int, np.int32, np.int64)
        and condition
    )


def int_2d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '2D array of integers'."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.dtype in (int, np.int32, np.int64)
        and condition
    )


def bool_1d(x: np.ndarray, condition: bool = True) -> bool:
    """Checks the variables is of type '1D array of booleans'."""
    return isinstance(x, np.ndarray) and x.ndim == 1 and x.dtype == bool and condition


def check_sim_complex_vecs4(
    a__: np.ndarray, b__: np.ndarray, c__: np.ndarray, d__: np.ndarray
) -> bool:
    """
    Checks similarity of 4 vectors, that is, all input vectors have same size,
    complex value type and are contiguous in memory.
    """
    return (
        isinstance(a__, np.ndarray)
        and isinstance(b__, np.ndarray)
        and isinstance(c__, np.ndarray)
        and isinstance(d__, np.ndarray)
        and a__.ndim == 1
        and a__.shape == b__.shape == c__.shape == d__.shape
        and a__.dtype == b__.dtype == c__.dtype == d__.dtype == np.cfloat
        and a__.data.contiguous
        and b__.data.contiguous
        and c__.data.contiguous
        and d__.data.contiguous
    )


def block_structure(num_qubits: int, blocks: np.ndarray) -> bool:
    """
    Checks validity of unit block structure.
    """
    return (
        isinstance(num_qubits, (int, np.int32, np.int64))
        and num_qubits >= 2
        and isinstance(blocks, np.ndarray)
        and blocks.dtype in (int, np.int32, np.int64)
        and blocks.shape == (2, blocks.size // 2)
        and np.all(np.logical_and(0 <= blocks, blocks < num_qubits))
        and np.all(blocks[0, :] != blocks[1, :])
    )


def check_permutation(x: np.ndarray) -> bool:
    """
    Checks if the input array is really an index permutation.
    """
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 1
        and x.dtype in (int, np.int32, np.int64)
        and np.all(np.sort(x) == np.arange(x.size, dtype=np.int64))
    )


def no_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Checks that two arrays do not overlap in memory.
    """
    return not np.may_share_memory(a, b)


def none_or_type(entity, entity_type) -> bool:
    """
    Checks that the first argument is None or has specified type.
    """
    return (entity is None) or isinstance(entity, entity_type)
