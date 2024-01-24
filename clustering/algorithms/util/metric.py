from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from ...util.numpy_extensions import (
    MAXIMUM_NDARRAY_BYTE_SIZE,
    convert_to_at_least_2d_tensor_inner,
    convert_to_inexact_numeric_tensor,
)

if TYPE_CHECKING:
    from ...util.typing_ import SimpleSequence

T = TypeVar('T')
T_contra = TypeVar('T_contra', contravariant=True)
NumericT = TypeVar('NumericT', int, float, complex, npt.NDArray[np.number])


@runtime_checkable
class ScalarDistanceFunction(Protocol[T_contra]):
    def __call__(
        self,
        element1: T_contra | npt.NDArray,
        element2: T_contra | npt.NDArray,
    ) -> int | float:
        ...


@runtime_checkable
class VectorizedDistanceFunction(Protocol[T_contra]):
    def __call__(
        self,
        element1: T_contra | npt.NDArray,
        element2: T_contra | npt.NDArray,
        _pairwise: bool = False,
    ) -> float | npt.NDArray[np.floating]:
        ...


DistanceFunction: TypeAlias = (
    ScalarDistanceFunction | VectorizedDistanceFunction
)


class QuasiMetric(ABC, Generic[T]):
    @abstractmethod
    def distance(
        self, element1: T, element2: T, _pairwise: bool = False
    ) -> int | float | npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        ...

    @abstractmethod
    def pairwise_distances(
        self,
        elements1: SimpleSequence[T],
        elements2: SimpleSequence[T],
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        ...


class ScalarQuasiMetric(QuasiMetric[T]):
    @abstractmethod
    def distance(
        self, element1: T, element2: T, _pairwise: bool = False
    ) -> int | float:
        ...

    def pairwise_distances(
        self,
        elements1: SimpleSequence[T],
        elements2: SimpleSequence[T],
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        elements1_size, elements2_size = len(elements1), len(elements2)
        pairwise_distances_matrix = np.zeros((elements1_size, elements2_size))
        for i in range(elements1_size):
            for j in range(elements2_size):
                pairwise_distances_matrix[i, j] = self.distance(
                    elements1[i], elements2[j]
                )
        return pairwise_distances_matrix


class DynamicScalarQuasiMetric(ScalarQuasiMetric[T]):
    def __init__(
        self,
        distance_function: ScalarDistanceFunction,
    ):
        self._distance_function = distance_function

    def distance(
        self, element1: T, element2: T, _pairwise: bool = False
    ) -> int | float:
        return self._distance_function(element1, element2)


class TensorQuasiMetric(QuasiMetric[NumericT]):
    @abstractmethod
    def distance(
        self,
        element1: NumericT | npt.NDArray[np.number],
        element2: NumericT | npt.NDArray[np.number],
        _pairwise: bool = False,
    ) -> float | npt.NDArray[np.floating]:
        ...

    def pairwise_distances_no_memory_limit(
        self,
        elements1: SimpleSequence[NumericT],
        elements2: SimpleSequence[NumericT],
    ) -> npt.NDArray[np.floating]:
        elements1_tensor = convert_to_at_least_2d_tensor_inner(elements1)
        elements2_tensor = convert_to_at_least_2d_tensor_inner(elements2)

        row_indexes_matrix, column_indexes_matrix = np.indices(
            (elements1_tensor.shape[0], elements2_tensor.shape[0])
        )
        row_indexes_array = row_indexes_matrix.ravel()
        column_indexes_array = column_indexes_matrix.ravel()

        pairwise_distances_array = self.distance(
            elements1_tensor[row_indexes_array],
            elements2_tensor[column_indexes_array],
            _pairwise=True,
        )

        pairwise_distances_matrix = np.zeros(
            (elements1_tensor.shape[0], elements2_tensor.shape[0]), dtype=float
        )
        pairwise_distances_matrix[
            row_indexes_array, column_indexes_array
        ] = pairwise_distances_array

        return pairwise_distances_matrix

    def pairwise_distances(
        self,
        elements1: SimpleSequence[NumericT],
        elements2: SimpleSequence[NumericT],
    ) -> npt.NDArray[np.floating]:
        elements1_tensor = convert_to_at_least_2d_tensor_inner(elements1)
        elements2_tensor = convert_to_at_least_2d_tensor_inner(elements2)

        row_indexes_matrix, column_indexes_matrix = np.indices(
            (elements1_tensor.shape[0], elements2_tensor.shape[0])
        )
        row_indexes_array = row_indexes_matrix.ravel()
        column_indexes_array = column_indexes_matrix.ravel()

        indexes_size = row_indexes_array.shape[0]
        pairwise_distances_array = np.zeros((indexes_size,), dtype=float)
        maximum_elements_subtensor_size = int(
            (MAXIMUM_NDARRAY_BYTE_SIZE / elements1_tensor[0].nbytes)
        )
        for i in range(0, indexes_size, maximum_elements_subtensor_size):
            row_indexes_slice = row_indexes_array[
                i : i + maximum_elements_subtensor_size
            ]
            column_indexes_slice = column_indexes_array[
                i : i + maximum_elements_subtensor_size
            ]
            pairwise_distances_array[
                i : i + maximum_elements_subtensor_size
            ] = self.distance(
                elements1_tensor[row_indexes_slice],
                elements2_tensor[column_indexes_slice],
                _pairwise=True,
            )

        pairwise_distances_matrix = np.zeros(
            (elements1_tensor.shape[0], elements2_tensor.shape[0]), dtype=float
        )
        pairwise_distances_matrix[
            row_indexes_array, column_indexes_array
        ] = pairwise_distances_array

        return pairwise_distances_matrix


class DynamicTensorQuasiMetric(TensorQuasiMetric[NumericT]):
    def __init__(
        self,
        vectorized_distance_function: VectorizedDistanceFunction,
    ):
        self._vectorized_distance_function_distance_function = (
            vectorized_distance_function
        )

    def distance(
        self,
        element1: NumericT | npt.NDArray[np.number],
        element2: NumericT | npt.NDArray[np.number],
        _pairwise: bool = False,
    ) -> float | npt.NDArray[np.floating]:
        return self._vectorized_distance_function_distance_function(
            element1, element2, _pairwise
        )


class TensorMetric(TensorQuasiMetric[NumericT]):
    @abstractmethod
    def norm(
        self,
        element: NumericT | npt.NDArray[np.number],
        _multiple: bool = False,
    ) -> float | npt.NDArray[np.floating]:
        ...

    def distance(
        self,
        element1: NumericT | npt.NDArray[np.number],
        element2: NumericT | npt.NDArray[np.number],
        _pairwise: bool = False,
    ) -> int | float | npt.NDArray[np.floating]:
        element1_tensor = convert_to_inexact_numeric_tensor(element1)
        element2_tensor = convert_to_inexact_numeric_tensor(element2)
        return self.norm(
            element1_tensor - element2_tensor, _multiple=_pairwise
        )


# https://en.wikipedia.org/wiki/Minkowski_distance
class MinkowskiMetric(TensorMetric[NumericT]):
    def __init__(self, order: int):
        self._order = order

    def norm(
        self,
        element: NumericT | npt.NDArray[np.number],
        _multiple: bool = False,
    ) -> float | npt.NDArray[np.floating]:
        element_tensor = convert_to_inexact_numeric_tensor(element)
        element_tensor_modes = tuple(range(element_tensor.ndim))
        if _multiple:
            sum_modes = element_tensor_modes[1:]
        else:
            sum_modes = element_tensor_modes
        element_norm_before_root: float | npt.NDArray[np.floating] = np.sum(
            np.abs(element_tensor) ** self._order, axis=sum_modes
        )
        element_norm: float | npt.NDArray[np.floating]
        if self._order == 1:
            element_norm = element_norm_before_root
        else:
            element_norm = element_norm_before_root ** (1 / self._order)
        return element_norm


# https://en.wikipedia.org/wiki/Taxicab_geometry
MANHATTAN_DISTANCE = MinkowskiMetric(1)

# https://en.wikipedia.org/wiki/Euclidean_distance
EUCLIDEAN_METRIC = MinkowskiMetric(2)


METRICS = dict(
    {
        'manhattan': MANHATTAN_DISTANCE,
        'euclidean': EUCLIDEAN_METRIC,
    }
)


def get_tensor_metric_from_name(
    metric_name: str,
) -> TensorMetric:
    if metric_name not in METRICS:
        raise ValueError(f'Unknown metric "{metric_name}".')
    return METRICS[metric_name]


def get_scalar_quasi_metric_from_distance_function(
    distance_function: ScalarDistanceFunction,
) -> QuasiMetric:
    return DynamicScalarQuasiMetric(distance_function)


def get_tensor_quasi_metric_from_distance_function(
    vectorized_distance_function: VectorizedDistanceFunction,
) -> TensorQuasiMetric:
    return DynamicTensorQuasiMetric(vectorized_distance_function)


def get_tensor_quasi_metric(
    metric_name_or_vectorized_distance_function: str
    | VectorizedDistanceFunction,
) -> TensorQuasiMetric:
    if isinstance(metric_name_or_vectorized_distance_function, str):
        return get_tensor_metric_from_name(
            metric_name_or_vectorized_distance_function
        )
    return get_tensor_quasi_metric_from_distance_function(
        metric_name_or_vectorized_distance_function
    )


def is_vectorized_distance_function(
    distance_function: DistanceFunction,
    element: T,
) -> bool:
    elements_list: list = [element, element]
    elements = convert_to_at_least_2d_tensor_inner(elements_list)
    if isinstance(distance_function, VectorizedDistanceFunction):
        try:
            distance_function(
                elements[:, np.newaxis, :],
                elements[np.newaxis, :, :],
                _pairwise=True,
            )
            return True
        except:  # pylint: disable=bare-except
            pass
    return False
