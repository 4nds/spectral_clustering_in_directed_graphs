import math
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from typing_extensions import override


class Kernel(ABC):
    @abstractmethod
    def get_kernel_matrix(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        ...


class GaussianKernel(Kernel):
    @abstractmethod
    def _get_kernel_width(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> int | float | npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        ...

    def get_kernel_matrix(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        kernel_width = self._get_kernel_width(distances_matrix)
        kernel_matrix: npt.NDArray[np.floating] = np.exp(
            -np.square(distances_matrix) / (2 * np.square(kernel_width))
        )
        return kernel_matrix


class ConstantWidthGaussianKernel(GaussianKernel):
    def __init__(
        self, kernel_width_as_percentage_of_maximum_distance: float = 0.1
    ):
        self._kernel_width_percentage = (
            kernel_width_as_percentage_of_maximum_distance
        )

    def _get_kernel_width(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> int | float | npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        kernel_width: float = (
            np.max(distances_matrix) * self._kernel_width_percentage
        )
        return kernel_width


class LocalConstantWidthGaussianKernel(ConstantWidthGaussianKernel):
    def __init__(
        self,
        kernel_width_as_percentage_of_maximum_distance: float = 0.1,
        neighborhood_radius_as_percentage_of_maximum_distance: float = 0.1,
    ):
        super().__init__(kernel_width_as_percentage_of_maximum_distance)
        self._neighborhood_radius_percentage = (
            neighborhood_radius_as_percentage_of_maximum_distance
        )

    @override
    def get_kernel_matrix(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        kernel_width = self._get_kernel_width(distances_matrix)
        kernel_matrix = np.zeros(distances_matrix.shape, dtype=float)
        neighborhood_radius = (
            np.max(distances_matrix) * self._neighborhood_radius_percentage
        )
        kernel_matrix_local_mask = distances_matrix < neighborhood_radius
        kernel_matrix[kernel_matrix_local_mask] = np.exp(
            -np.square(distances_matrix[kernel_matrix_local_mask])
            / (2 * np.square(kernel_width))
        )
        return kernel_matrix


class ImageGaussianKernel(Kernel):
    @abstractmethod
    def _get_kernel_width(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> int | float | npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        ...

    def get_kernel_matrix(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        kernel_width = self._get_kernel_width(distances_matrix)
        kernel_matrix: npt.NDArray[np.floating] = np.exp(
            -np.square(distances_matrix) / kernel_width
        )
        return kernel_matrix


class ConstantWidthImageGaussianKernel(ImageGaussianKernel):
    def __init__(self, kernel_width: float):
        self._kernel_width = kernel_width

    def _get_kernel_width(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> int | float | npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        return self._kernel_width


class LocalConstantWidthImageGaussianKernel(ConstantWidthImageGaussianKernel):
    def __init__(
        self,
        kernel_width: float,
        neighborhood_radius: float,
    ):
        super().__init__(kernel_width)
        self._neighborhood_radius = neighborhood_radius

    @override
    def get_kernel_matrix(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        kernel_width = self._get_kernel_width(distances_matrix)
        kernel_matrix = np.zeros(distances_matrix.shape, dtype=float)
        kernel_matrix_local_mask = distances_matrix < self._neighborhood_radius
        kernel_matrix[kernel_matrix_local_mask] = np.exp(
            -np.square(distances_matrix[kernel_matrix_local_mask])
            / kernel_width
        )
        return kernel_matrix


class DynamicWidthGaussianKernel(GaussianKernel):
    def __init__(
        self,
        kernel_width_as_percentage_of_maximum_distance: float = 0.1,
        number_of_neighbors_factor: float = 1,
    ):
        self._kernel_width_percentage = (
            kernel_width_as_percentage_of_maximum_distance
        )
        self._number_of_neighbors_factor = number_of_neighbors_factor

    def _get_kernel_width(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> int | float | npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        matrix_size = distances_matrix.shape[0]
        number_of_observed_neighbors = int(
            math.isqrt(matrix_size) * self._number_of_neighbors_factor
        )
        partially_sorted_distances_matrix_indexes = np.argpartition(
            distances_matrix, number_of_observed_neighbors - 1, axis=0
        )
        indexes_of_farthest_neighbors = (
            partially_sorted_distances_matrix_indexes[
                number_of_observed_neighbors - 1, :
            ]
        )
        # maximum_distance_to_observed_neighbors: npt.NDArray[
        #     np.integer
        # ] | npt.NDArray[np.floating] = (
        #     distances_matrix[
        #         indexes_of_farthest_neighbors, np.arange(matrix_size)
        #     ]
        #     * self._kernel_width_percentage
        # )
        maximum_distance_to_observed_neighbors: npt.NDArray[
            np.integer
        ] | npt.NDArray[np.floating] = (
            distances_matrix[
                indexes_of_farthest_neighbors, np.arange(matrix_size)
            ]
            * np.max(distances_matrix)
            * self._kernel_width_percentage
        )
        return maximum_distance_to_observed_neighbors


class LocalDynamicWidthGaussianKernel(DynamicWidthGaussianKernel):
    def __init__(
        self,
        kernel_width_as_percentage_of_maximum_distance: float = 0.1,
        number_of_neighbors_factor: float = 1,
        neighborhood_radius_as_percentage_of_maximum_distance: float = 0.1,
    ):
        super().__init__(
            kernel_width_as_percentage_of_maximum_distance,
            number_of_neighbors_factor,
        )
        self._neighborhood_radius_percentage = (
            neighborhood_radius_as_percentage_of_maximum_distance
        )

    @override
    def get_kernel_matrix(
        self,
        distances_matrix: npt.NDArray[np.integer] | npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        kernel_width = self._get_kernel_width(distances_matrix)
        kernel_matrix = np.zeros(distances_matrix.shape, dtype=float)
        neighborhood_radius = (
            np.max(distances_matrix) * self._neighborhood_radius_percentage
        )
        kernel_matrix_local_mask = distances_matrix < neighborhood_radius
        kernel_matrix[kernel_matrix_local_mask] = np.exp(
            -np.square(distances_matrix) / (2 * np.square(kernel_width))
        )[kernel_matrix_local_mask]
        return kernel_matrix


CONSTANT_WIDTH_GAUSSIAN_KERNEL = ConstantWidthGaussianKernel(0.02)

DYNAMIC_WIDTH_GAUSSIAN_KERNEL = DynamicWidthGaussianKernel(1)
