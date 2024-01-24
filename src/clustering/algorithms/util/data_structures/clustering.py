from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

import cv2
import numpy as np
import numpy.typing as npt
from typing_extensions import override

from ....util.numpy_extensions import convert_to_at_least_2d_tensor_inner
from ..metric import DistanceFunction
from .data_set import DataSet, DataTypeBasedDataSet

if TYPE_CHECKING:
    from ....util.typing_ import SimpleSequence

T = TypeVar('T')


class Clustering(DataSet[T]):
    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @property
    @abstractmethod
    def current_size(self) -> int:
        ...

    @property
    @abstractmethod
    def labels(self) -> list[int]:
        ...

    @abstractmethod
    def get_cluster_size(self, index: int) -> int:
        ...

    @abstractmethod
    def get_cluster_elements(self, index: int) -> SimpleSequence[T]:
        ...

    @abstractmethod
    def get_clusters_indexes(self) -> list[int]:
        ...

    @abstractmethod
    def join_clusters(self, index1: int, index2: int) -> int:
        ...

    @abstractmethod
    def analyze(self) -> None:
        ...


class ArrayBasedClustering(Clustering[T], DataTypeBasedDataSet[T]):
    def __init__(
        self,
        data: SimpleSequence[T],
        size: int,
        metric: str | DistanceFunction = 'euclidean',
    ):
        super().__init__(data, metric)
        self._elements = np.asarray(super().elements)
        self._size = size
        self._current_size = self._elements.shape[0]
        self._subset_indexes = np.arange(self._current_size)

    @property
    @override
    def elements(self) -> npt.NDArray:
        return self._elements

    @property
    def size(self) -> int:
        return self._size

    @property
    def current_size(self) -> int:
        return self._current_size

    @property
    def subset_indexes(self) -> npt.NDArray[np.int_]:
        return self._subset_indexes

    def get_cluster_size(self, index: int) -> int:
        cluster_size: int = np.sum(self._subset_indexes == index)
        return cluster_size

    def get_cluster_elements(self, index: int) -> npt.NDArray:
        subset: npt.NDArray = self._elements[self._subset_indexes == index]
        return subset

    def get_clusters_indexes(self) -> list[int]:
        return sorted(set(self._subset_indexes))

    def join_clusters(self, index1: int, index2: int) -> int:
        min_index, max_index = min(index1, index2), max(index1, index2)
        self._subset_indexes[self._subset_indexes == max_index] = min_index
        self._current_size -= 1
        return min_index

    def rename_subset_indexes(self) -> None:
        old_labels: list[int] = self.subset_indexes.tolist()
        possible_labels = set(old_labels)
        old_to_new_labels = {
            label: label
            for label in possible_labels
            if label < len(possible_labels)
        }
        available_labels = set(range(len(possible_labels))) - set(
            old_to_new_labels.keys()
        )
        new_labels: list[int] = []
        for label in old_labels:
            if label not in old_to_new_labels:
                old_to_new_labels[label] = available_labels.pop()
            new_labels.append(old_to_new_labels[label])
        self._subset_indexes = np.asarray(new_labels)


class ImagePixelsClustering(ArrayBasedClustering[npt.NDArray[np.int_]]):
    def __init__(
        self,
        image_data: SimpleSequence[npt.NDArray[np.int_]],
        size: int,
        metric: str | DistanceFunction = 'euclidean',
    ):
        super().__init__(image_data, size, metric)

        self._image_intensity: npt.NDArray[np.number] = cv2.cvtColor(
            image_data, cv2.COLOR_RGB2GRAY
        )
        self._image_intensity = self._image_intensity.astype(float) / 255

        # hsv_image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
        # # pylint: disable=invalid-name
        # h = hsv_image_data[:, :, 0]
        # s = hsv_image_data[:, :, 1]
        # v = hsv_image_data[:, :, 2]
        # # pylint: enable=invalid-name
        # self._image_intensity = np.stack(
        #     (v, v * s * np.sin(h), v * s * np.cos(h)), axis=2
        # )

        image_height, image_width = self._image_intensity.shape[:2]
        row_coordinates, column_coordinates = np.meshgrid(
            # np.arange(image_width), np.arange(image_height), indexing='ij'
            np.arange(image_width),
            np.arange(image_height),
            indexing='xy',
        )
        row_coordinates = convert_to_at_least_2d_tensor_inner(
            row_coordinates.ravel()
        )
        column_coordinates = convert_to_at_least_2d_tensor_inner(
            column_coordinates.ravel()
        )
        self._image_spatial_location: npt.NDArray[np.int_] = np.concatenate(
            (row_coordinates, column_coordinates), axis=1
        )

        # self._intensity_distances_matrix = self.metric.pairwise_distances(
        #     self._image_intensity.reshape(-1, 3),
        #     self._image_intensity.reshape(-1, 3),
        # )

        self._intensity_distances_matrix = self.metric.pairwise_distances(
            self._image_intensity.reshape(-1, 1),
            self._image_intensity.reshape(-1, 1),
        )

        self._spatial_location_distances_matrix = (
            self.metric.pairwise_distances(
                self._image_spatial_location, self._image_spatial_location
            )
        )

    @property
    @abstractmethod
    def image_shaped_labels(self) -> npt.NDArray[np.int_]:
        ...

    @property
    def image_intensity(self) -> npt.NDArray[np.number]:
        return self._image_intensity

    @property
    def image_spatial_location(self) -> npt.NDArray[np.int_]:
        return self._image_spatial_location

    @property
    def intensity_distances_matrix(
        self,
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        return self._intensity_distances_matrix

    @property
    def spatial_location_distances_matrix(
        self,
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        return self._spatial_location_distances_matrix
