from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Tuple,
    TypeVar,
    cast,
    overload,
)

import numpy as np
import numpy.typing as npt

from ..util.typing_ import MaskedNDArray, NumpyGenericLike, NumpyGenericLikeT

if TYPE_CHECKING:
    from ..util.typing_ import SimpleSequence

T = TypeVar('T')


MAXIMUM_NDARRAY_BYTE_SIZE = 10**8


def convert_to_at_least_2d_tensor_inner(
    sequence: SimpleSequence,
) -> npt.NDArray:
    tensor = np.asarray(sequence)
    new_shape = tensor.shape + (1,) * (2 - tensor.ndim)
    return tensor.reshape(new_shape)


def convert_to_at_least_2d_tensor_outer(
    sequence: SimpleSequence,
) -> npt.NDArray:
    tensor = np.asarray(sequence)
    new_shape = (1,) * (2 - tensor.ndim) + tensor.shape
    return tensor.reshape(new_shape)


def get_indexes_of_min_element_in_tensor(
    tensor: npt.NDArray,
) -> Tuple[int, ...]:
    min_flatten_index = np.argmin(tensor)
    min_numpy_integer_indexes = np.unravel_index(
        min_flatten_index, tensor.shape
    )
    min_indexes = tuple(map(int, min_numpy_integer_indexes))
    return min_indexes


def delete_index_in_tensor_unoptimized_compress(
    tensor: npt.NDArray, index: int, mode: int
) -> npt.NDArray:
    mask = np.ones(tensor.shape[mode], dtype=bool)
    mask[index] = False
    tensor_without_index: npt.NDArray = np.compress(mask, tensor, axis=mode)
    return tensor_without_index


def delete_index_in_tensor_unoptimized_indexing(
    tensor: npt.NDArray, index: int, mode: int
) -> npt.NDArray:
    mask = np.ones(tensor.shape[mode], dtype=bool)
    mask[index] = False
    tensor_without_index = tensor[(slice(None),) * mode + (mask,)]
    return tensor_without_index


def get_smallest_subtensor_from_masked_tensor(
    tensor: npt.NDArray,
) -> MaskedNDArray:
    masked_tensor: MaskedNDArray = np.ma.asarray(tensor)
    modes = tuple(range(masked_tensor.ndim))
    not_masked = ~masked_tensor.mask
    if len(not_masked.shape) == 0 and bool(not_masked) is True:
        return masked_tensor
    for mode in modes:
        all_modes_except_one = tuple(m for m in modes if m != mode)
        selection_mask = np.any(not_masked, axis=all_modes_except_one)
        if not np.all(selection_mask):
            masked_tensor = np.compress(
                selection_mask, masked_tensor, axis=mode
            )
    return masked_tensor


def delete_index_in_tensor(
    tensor: npt.NDArray,
    index: int,
    mode: int,
    maximum_masked_to_data_ratio: float = 0.5,
) -> MaskedNDArray:
    masked_tensor: MaskedNDArray = np.ma.asarray(tensor)
    masked_tensor[(slice(None),) * mode + (index,)] = np.ma.masked
    masked_to_data_ratio = np.sum(masked_tensor.mask) / masked_tensor.size
    if masked_to_data_ratio < maximum_masked_to_data_ratio:
        return masked_tensor
    return get_smallest_subtensor_from_masked_tensor(masked_tensor)


def get_index_of_element_in_vector(
    element: Any,
    tensor: npt.NDArray,
    default_index_value: int | T = -1,
) -> int | T:
    element_indexes = np.argwhere(tensor == element)
    if element_indexes.size == 0:
        return default_index_value
    element_index: int = element_indexes[0, 0]
    return element_index


def convert_to_inexact_numeric_tensor(
    element: npt.ArrayLike,
) -> npt.NDArray[np.floating] | npt.NDArray[np.complexfloating]:
    element_tensor = np.asarray(element)
    if issubclass(element_tensor.dtype.type, np.inexact):
        return element_tensor
    return element_tensor.astype(float)


def _apply_function_to_tensor_element(
    function: Callable[[NumpyGenericLikeT], NumpyGenericLike]
    | Callable[[NumpyGenericLikeT, Tuple[int, ...]], NumpyGenericLike],
    element: NumpyGenericLikeT,
    multi_index: Tuple[int, ...],
    use_multi_index: bool = False,
) -> Any:
    if use_multi_index:
        function = cast(
            Callable[[NumpyGenericLikeT, Tuple[int, ...]], NumpyGenericLike],
            function,
        )
        return function(element, multi_index)
    else:
        function = cast(
            Callable[[NumpyGenericLikeT], NumpyGenericLike], function
        )
        return function(element)


@overload
def tensor_map(
    function: Callable[[NumpyGenericLikeT], NumpyGenericLike],
    tensor: npt.NDArray,
    use_multi_index: Literal[False] = False,
) -> npt.NDArray:
    ...


@overload
def tensor_map(
    function: Callable[[NumpyGenericLikeT, Tuple[int, ...]], NumpyGenericLike],
    tensor: npt.NDArray,
    use_multi_index: Literal[True],
) -> npt.NDArray:
    ...


def tensor_map(
    function: Callable[[NumpyGenericLikeT], NumpyGenericLike]
    | Callable[[NumpyGenericLikeT, Tuple[int, ...]], NumpyGenericLike],
    tensor: npt.NDArray,
    use_multi_index: bool = False,
) -> npt.NDArray:
    return np.array(
        [
            _apply_function_to_tensor_element(
                function, element, multi_index, use_multi_index
            )
            for multi_index, element in np.ndenumerate(tensor)
        ]
    )


@overload
def masked_tensor_map(
    function: Callable[[NumpyGenericLikeT], NumpyGenericLike],
    masked_tensor: MaskedNDArray,
    use_multi_index: Literal[False] = False,
) -> MaskedNDArray:
    ...


@overload
def masked_tensor_map(
    function: Callable[[NumpyGenericLikeT, Tuple[int, ...]], NumpyGenericLike],
    masked_tensor: MaskedNDArray,
    use_multi_index: Literal[True],
) -> MaskedNDArray:
    ...


def masked_tensor_map(
    function: Callable[[NumpyGenericLikeT], NumpyGenericLike]
    | Callable[[NumpyGenericLikeT, Tuple[int, ...]], NumpyGenericLike],
    masked_tensor: MaskedNDArray,
    use_multi_index: bool = False,
) -> MaskedNDArray:
    mapped_non_masked_elements = np.array(
        [
            _apply_function_to_tensor_element(
                function, element, multi_index, use_multi_index
            )
            for multi_index, element in np.ma.ndenumerate(masked_tensor)
        ]
    )
    masked_mapped_tensor: MaskedNDArray = np.ma.empty(
        masked_tensor.shape, dtype=mapped_non_masked_elements.dtype
    )
    masked_mapped_tensor.mask = masked_tensor.mask
    np.place(
        masked_mapped_tensor,
        ~masked_mapped_tensor.mask,
        mapped_non_masked_elements,
    )
    return masked_mapped_tensor
