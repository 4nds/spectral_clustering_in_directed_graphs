from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import numpy.typing as npt
import scipy

from ..util.data_structures.clustering import (
    ArrayBasedClustering,
    ImagePixelsClustering,
)
from ..util.kernel import (
    CONSTANT_WIDTH_GAUSSIAN_KERNEL,
    DYNAMIC_WIDTH_GAUSSIAN_KERNEL,
    Kernel,
)
from ..util.metric import DistanceFunction
from .k_means_clustering import get_k_means_clustering

if TYPE_CHECKING:
    from ...util.typing_ import SimpleSequence

T = TypeVar('T')


class UndirectedSpectralClusteringBase(ArrayBasedClustering[T]):
    @abstractmethod
    def get_weight_matrix(self) -> npt.NDArray[np.floating]:
        ...

    def get_clustering_labels(self) -> list[int]:
        weight_matrix = self.get_weight_matrix()
        degree_matrix = np.diag(np.sum(weight_matrix, axis=1))
        laplacian_matrix = degree_matrix - weight_matrix
        # Degree matrix is diagonal so we can use element-wise square root.
        inverse_of_sqrt_of_degree_matrix = np.linalg.inv(
            np.sqrt(degree_matrix)
        )
        normalized_laplacian_matrix = (
            inverse_of_sqrt_of_degree_matrix
            @ laplacian_matrix
            @ inverse_of_sqrt_of_degree_matrix
        )
        _eigenvalues, eigenvectors_matrix = scipy.linalg.eigh(
            normalized_laplacian_matrix,
            subset_by_index=(0, self.size - 1),
        )
        relaxed_solution = (
            inverse_of_sqrt_of_degree_matrix @ eigenvectors_matrix
        )
        k_means_clustering_of_rows_in_relaxed_solution = (
            get_k_means_clustering(relaxed_solution, self.size)
        )
        clustering_labels = (
            k_means_clustering_of_rows_in_relaxed_solution.labels
        )
        return clustering_labels


class UndirectedSpectralClustering(UndirectedSpectralClusteringBase[T]):
    def __init__(
        self,
        data: SimpleSequence[T],
        size: int,
        metric: str | DistanceFunction = 'euclidean',
        kernel: Kernel = CONSTANT_WIDTH_GAUSSIAN_KERNEL,
    ):
        super().__init__(data, size, metric)
        self._kernel = kernel
        self._labels: list[int] = []

    @property
    def labels(self) -> list[int]:
        return self._labels

    def get_weight_matrix(self) -> npt.NDArray[np.floating]:
        weight_matrix = self._kernel.get_kernel_matrix(
            self.element_distances_matrix
        )
        return weight_matrix

    def analyze(self) -> None:
        self._labels = self.get_clustering_labels()


class ImageUndirectedSpectralClustering(
    UndirectedSpectralClusteringBase[npt.NDArray[np.int_]],
    ImagePixelsClustering,
):
    def __init__(
        self,
        image_data: SimpleSequence[npt.NDArray[np.int_]],
        size: int,
        metric: str | DistanceFunction = 'euclidean',
        intensity_kernel: Kernel = CONSTANT_WIDTH_GAUSSIAN_KERNEL,
        spatial_location_kernel: Kernel = CONSTANT_WIDTH_GAUSSIAN_KERNEL,
    ):
        super().__init__(
            image_data,
            size,
            metric,
        )
        self._intensity_kernel = intensity_kernel
        self._spatial_location_kernel = spatial_location_kernel
        self._labels: list[int] = []
        self._image_shaped_labels = np.array([], dtype=int)

    @property
    def labels(self) -> list[int]:
        return self._labels

    @property
    def image_shaped_labels(self) -> npt.NDArray[np.int_]:
        return self._image_shaped_labels

    def get_weight_matrix(self) -> npt.NDArray[np.floating]:
        weight_matrix = self._intensity_kernel.get_kernel_matrix(
            self.intensity_distances_matrix
        )
        # weight_matrix = self._spatial_location_kernel.get_kernel_matrix(
        #     self.spatial_location_distances_matrix
        # )
        # weight_matrix = self._intensity_kernel.get_kernel_matrix(
        #     self.intensity_distances_matrix
        # ) * self._spatial_location_kernel.get_kernel_matrix(
        #     self.spatial_location_distances_matrix
        # )
        return weight_matrix

    def analyze(self) -> None:
        self._labels = self.get_clustering_labels()
        flattened_clustering_labels = np.array(self._labels)
        image_height, image_width = self.elements.shape[:2]
        self._image_shaped_labels = flattened_clustering_labels.reshape(
            (image_height, image_width)
        )


class DirectedSpectralClusteringBase(ArrayBasedClustering[T]):
    @abstractmethod
    def get_weight_matrix(self) -> npt.NDArray[np.floating]:
        ...

    def get_clustering_labels(self) -> list[int]:
        weight_matrix = self.get_weight_matrix()
        degree_matrix = np.diag(np.sum(weight_matrix, axis=1))
        laplacian_matrix = degree_matrix - weight_matrix
        # Degree matrix is diagonal so we can use element-wise square root.
        inverse_of_sqrt_of_degree_matrix = np.linalg.inv(
            np.sqrt(degree_matrix)
        )
        normalized_laplacian_matrix = (
            inverse_of_sqrt_of_degree_matrix
            @ laplacian_matrix
            @ inverse_of_sqrt_of_degree_matrix
        )
        hermitian_part_of_normalized_laplacian_matrix = (
            1
            / 2
            * (
                normalized_laplacian_matrix
                + np.transpose(normalized_laplacian_matrix)
            )
        )

        _eigenvalues, eigenvectors_matrix = scipy.linalg.eigh(
            hermitian_part_of_normalized_laplacian_matrix,
            subset_by_index=(0, self.size - 1),
        )
        relaxed_solution = (
            inverse_of_sqrt_of_degree_matrix @ eigenvectors_matrix
        )
        k_means_clustering_of_rows_in_relaxed_solution = (
            get_k_means_clustering(relaxed_solution, self.size)
        )
        clustering_labels = (
            k_means_clustering_of_rows_in_relaxed_solution.labels
        )
        return clustering_labels


class DirectedSpectralClustering(DirectedSpectralClusteringBase[T]):
    def __init__(
        self,
        data: SimpleSequence[T],
        size: int,
        metric: str | DistanceFunction = 'euclidean',
        kernel: Kernel = DYNAMIC_WIDTH_GAUSSIAN_KERNEL,
    ):
        super().__init__(data, size, metric)
        self._kernel = kernel
        self._labels: list[int] = []

    @property
    def labels(self) -> list[int]:
        return self._labels

    def get_weight_matrix(self) -> npt.NDArray[np.floating]:
        weight_matrix = self._kernel.get_kernel_matrix(
            self.element_distances_matrix
        )
        return weight_matrix

    def analyze(self) -> None:
        self._labels = self.get_clustering_labels()


class ImageDirectedSpectralClustering(
    DirectedSpectralClusteringBase[npt.NDArray[np.int_]],
    ImagePixelsClustering,
):
    def __init__(
        self,
        image_data: SimpleSequence[npt.NDArray[np.int_]],
        size: int,
        metric: str | DistanceFunction = 'euclidean',
        intensity_kernel: Kernel = DYNAMIC_WIDTH_GAUSSIAN_KERNEL,
        spatial_location_kernel: Kernel = DYNAMIC_WIDTH_GAUSSIAN_KERNEL,
    ):
        super().__init__(
            image_data,
            size,
            metric,
        )
        self._intensity_kernel = intensity_kernel
        self._spatial_location_kernel = spatial_location_kernel
        self._labels: list[int] = []
        self._image_shaped_labels = np.array([], dtype=int)

    @property
    def labels(self) -> list[int]:
        return self._labels

    @property
    def image_shaped_labels(self) -> npt.NDArray[np.int_]:
        return self._image_shaped_labels

    def get_weight_matrix(self) -> npt.NDArray[np.floating]:
        # weight_matrix = self._intensity_kernel.get_kernel_matrix(
        #     self.intensity_distances_matrix
        # )
        # weight_matrix = self._spatial_location_kernel.get_kernel_matrix(
        #     self.spatial_location_distances_matrix
        # )
        weight_matrix = self._intensity_kernel.get_kernel_matrix(
            self.intensity_distances_matrix
        ) * self._spatial_location_kernel.get_kernel_matrix(
            self.spatial_location_distances_matrix
        )
        return weight_matrix

    def analyze(self) -> None:
        self._labels = self.get_clustering_labels()
        flattened_clustering_labels = np.array(self._labels)
        image_height, image_width = self.elements.shape[:2]
        self._image_shaped_labels = flattened_clustering_labels.reshape(
            (image_height, image_width)
        )


def get_undirected_spectral_clustering(
    data: SimpleSequence[T],
    number_of_clusters: int,
    metric: str | DistanceFunction = 'euclidean',
    kernel: Kernel = CONSTANT_WIDTH_GAUSSIAN_KERNEL,
) -> UndirectedSpectralClustering:
    undirected_spectral_clustering = UndirectedSpectralClustering(
        data, number_of_clusters, metric, kernel
    )
    undirected_spectral_clustering.analyze()
    return undirected_spectral_clustering


def get_directed_spectral_clustering(
    data: SimpleSequence[T],
    number_of_clusters: int,
    metric: str | DistanceFunction = 'euclidean',
    kernel: Kernel = DYNAMIC_WIDTH_GAUSSIAN_KERNEL,
) -> DirectedSpectralClustering:
    directed_spectral_clustering = DirectedSpectralClustering(
        data, number_of_clusters, metric, kernel
    )
    directed_spectral_clustering.analyze()
    return directed_spectral_clustering
