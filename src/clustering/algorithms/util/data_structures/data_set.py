from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from ..metric import (
    DistanceFunction,
    QuasiMetric,
    ScalarDistanceFunction,
    VectorizedDistanceFunction,
    get_scalar_quasi_metric_from_distance_function,
    get_tensor_quasi_metric,
    is_vectorized_distance_function,
)

if TYPE_CHECKING:
    from ....util.typing_ import SimpleSequence

T = TypeVar('T')


class DataSet(ABC, Generic[T]):
    @property
    @abstractmethod
    def elements(self) -> SimpleSequence[T]:
        ...

    @property
    @abstractmethod
    def element_distances_matrix(
        self,
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        ...


class MetricDataSet(DataSet[T]):
    @property
    @abstractmethod
    def metric(self) -> QuasiMetric[T]:
        ...


class SequenceBasedDataSet(MetricDataSet[T]):
    def __init__(
        self,
        data: SimpleSequence[T],
        metric: ScalarDistanceFunction,
    ):
        self._elements = data
        self._metric = get_scalar_quasi_metric_from_distance_function(metric)
        self._element_distances_matrix = self._metric.pairwise_distances(
            self._elements, self._elements
        )

    @property
    def elements(self) -> SimpleSequence[T]:
        return self._elements

    @property
    def element_distances_matrix(
        self,
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        return self._element_distances_matrix

    @property
    def metric(self) -> QuasiMetric[T]:
        return self._metric


class TensorBasedDataSet(MetricDataSet[T]):
    def __init__(
        self,
        data: SimpleSequence[T],
        tensor_metric: str | VectorizedDistanceFunction = 'euclidean',
    ):
        self._elements = np.asarray(data)
        self._tensor_metric = get_tensor_quasi_metric(tensor_metric)
        self._element_distances_matrix = (
            self._tensor_metric.pairwise_distances(
                self._elements, self._elements
            )
        )

    @property
    def elements(self) -> npt.NDArray:
        return self._elements

    @property
    def element_distances_matrix(
        self,
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        return self._element_distances_matrix

    @property
    def metric(self) -> QuasiMetric[T]:
        return self._tensor_metric


class DataTypeBasedDataSet(MetricDataSet[T]):
    def __init__(
        self,
        data: SimpleSequence[T],
        metric: str | DistanceFunction = 'euclidean',
    ):
        self._data_set_object = DataTypeBasedDataSet._get_data_set_object(
            data, metric
        )
        return

    @staticmethod
    def _get_data_set_object(
        data: SimpleSequence[T],
        metric: str | DistanceFunction,
    ) -> MetricDataSet[T]:
        if isinstance(metric, str):
            return TensorBasedDataSet(data, metric)
        if is_vectorized_distance_function(metric, data[0]):
            assert isinstance(metric, VectorizedDistanceFunction)
            return TensorBasedDataSet(data, metric)
        assert isinstance(metric, ScalarDistanceFunction)
        return SequenceBasedDataSet(data, metric)

    @property
    def elements(self) -> SimpleSequence[T]:
        return self._data_set_object.elements

    @property
    def element_distances_matrix(
        self,
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        return self._data_set_object.element_distances_matrix

    @property
    def metric(self) -> QuasiMetric[T]:
        return self._data_set_object.metric
