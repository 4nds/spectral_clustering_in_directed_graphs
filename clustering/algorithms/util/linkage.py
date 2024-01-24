from __future__ import annotations

import math
import operator
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Callable,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from ...util.numpy_extensions import (
    delete_index_in_tensor,
    get_indexes_of_min_element_in_tensor,
    masked_tensor_map,
)
from ...util.typing_ import MaskedNDArray
from .data_structures.clustering import Clustering
from .metric import QuasiMetric

if TYPE_CHECKING:
    from ...util.typing_ import SimpleSequence

T = TypeVar('T')
NumberT = TypeVar('NumberT', int, float, complex)


#
#
# Linkage classes


class Linkage(ABC, Generic[T]):
    @property
    @abstractmethod
    def metric(self) -> QuasiMetric[T]:
        ...

    @abstractmethod
    def distance(
        self, cluster1: SimpleSequence[T], cluster2: SimpleSequence[T]
    ) -> int | float:
        ...


class StaticLinkage(Linkage[T]):
    def __init__(self, metric: QuasiMetric[T]):
        self._metric = metric

    @property
    def metric(self) -> QuasiMetric[T]:
        return self._metric


class SequenceRecursiveLinkage(Linkage[T]):
    @abstractmethod
    def joined_cluster_distances(
        self,
        old_cluster1_distances: SimpleSequence[int | float],
        cluster1_position: int,
        old_cluster2_distances: SimpleSequence[int | float],
        cluster2_position: int,
        old_cluster_sizes: SimpleSequence[int],
    ) -> list[int | float]:
        ...


class MaskedArrayRecursiveLinkage(Linkage[T]):
    @abstractmethod
    def joined_cluster_distances(
        self,
        old_cluster1_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster1_position: int,
        old_cluster2_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster2_position: int,
        old_cluster_sizes: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
    ) -> MaskedNDArray[np.integer] | MaskedNDArray[np.floating]:
        ...


# https://en.wikipedia.org/wiki/Complete-linkage_clustering
class CompleteLinkage(StaticLinkage[T], MaskedArrayRecursiveLinkage[T]):
    def distance(
        self, cluster1: SimpleSequence[T], cluster2: SimpleSequence[T]
    ) -> int | float:
        element_distances_matrix = self.metric.pairwise_distances(
            cluster1, cluster2
        )
        clusters_distance = np.max(element_distances_matrix)
        return cast(int | float, clusters_distance)

    def joined_cluster_distances(
        self,
        old_cluster1_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster1_position: int,
        old_cluster2_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster2_position: int,
        old_cluster_sizes: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
    ) -> MaskedNDArray[np.integer] | MaskedNDArray[np.floating]:
        union_distances: MaskedNDArray[np.integer] | MaskedNDArray[np.floating]
        union_distances = np.ma.max(
            [
                old_cluster1_distances,
                old_cluster2_distances,
            ],
            axis=0,
        )
        return union_distances


# https://en.wikipedia.org/wiki/Single-linkage_clustering
class SingleLinkage(StaticLinkage[T], MaskedArrayRecursiveLinkage[T]):
    def distance(
        self, cluster1: SimpleSequence[T], cluster2: SimpleSequence[T]
    ) -> int | float:
        element_distances_matrix = self.metric.pairwise_distances(
            cluster1, cluster2
        )
        clusters_distance = np.min(element_distances_matrix)
        return cast(int | float, clusters_distance)

    def joined_cluster_distances(
        self,
        old_cluster1_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster1_position: int,
        old_cluster2_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster2_position: int,
        old_cluster_sizes: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
    ) -> MaskedNDArray[np.integer] | MaskedNDArray[np.floating]:
        union_distances: MaskedNDArray[np.integer] | MaskedNDArray[np.floating]
        union_distances = np.ma.min(
            [
                old_cluster1_distances,
                old_cluster2_distances,
            ],
            axis=0,
        )
        return union_distances


# https://en.wikipedia.org/wiki/UPGMA
class AverageLinkage(StaticLinkage[T], MaskedArrayRecursiveLinkage[T]):
    def distance(
        self, cluster1: SimpleSequence[T], cluster2: SimpleSequence[T]
    ) -> int | float:
        element_distances_matrix = self.metric.pairwise_distances(
            cluster1, cluster2
        )
        clusters_distance = np.sum(element_distances_matrix) / (
            len(cluster1) * len(cluster2)
        )
        return cast(int | float, clusters_distance)

    def joined_cluster_distances(
        self,
        old_cluster1_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster1_position: int,
        old_cluster2_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster2_position: int,
        old_cluster_sizes: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
    ) -> MaskedNDArray[np.integer] | MaskedNDArray[np.floating]:
        old_cluster1_size = old_cluster_sizes[cluster1_position]
        old_cluster2_size = old_cluster_sizes[cluster2_position]
        union_distances: MaskedNDArray[np.integer] | MaskedNDArray[np.floating]
        union_distances = (
            old_cluster1_distances * old_cluster1_size
            + old_cluster2_distances * old_cluster2_size
        ) / (old_cluster1_size + old_cluster2_size)
        return union_distances


# https://en.wikipedia.org/wiki/Hierarchical_clustering#Cluster_Linkage
class CentroidLinkage(StaticLinkage[T], MaskedArrayRecursiveLinkage[T]):
    def distance(
        self,
        cluster1: SimpleSequence[T],
        cluster2: SimpleSequence[T],
    ) -> int | float:
        cluster1_tensor = np.asarray(cluster1)
        cluster2_tensor = np.asarray(cluster2)
        cluster1_centroid = np.mean(cluster1_tensor, axis=0)
        cluster2_centroid = np.mean(cluster2_tensor, axis=0)
        clusters_distance = self.metric.distance(
            cluster1_centroid, cluster2_centroid
        )
        return cast(int | float, clusters_distance)

    def joined_cluster_distances(
        self,
        old_cluster1_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster1_position: int,
        old_cluster2_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster2_position: int,
        old_cluster_sizes: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
    ) -> MaskedNDArray[np.integer] | MaskedNDArray[np.floating]:
        old_cluster1_size = old_cluster_sizes[cluster1_position]
        old_cluster2_size = old_cluster_sizes[cluster2_position]
        cluster_union_size = old_cluster1_size + old_cluster2_size
        old_cluster1_size_ratio = old_cluster1_size / cluster_union_size
        old_cluster2_size_ratio = old_cluster2_size / cluster_union_size
        cluster1_to_cluster2_distance = old_cluster1_distances[
            cluster2_position
        ]
        cluster2_to_cluster1_distance = old_cluster2_distances[
            cluster1_position
        ]

        union_distances: MaskedNDArray[np.integer] | MaskedNDArray[np.floating]
        # np.abs() is need to fix errors caused by floating point (im)precision
        union_distances = np.sqrt(
            np.abs(
                old_cluster1_size_ratio * np.square(old_cluster1_distances)
                + old_cluster2_size_ratio * np.square(old_cluster2_distances)
                - old_cluster1_size_ratio
                * old_cluster2_size_ratio
                * cluster1_to_cluster2_distance
                * cluster2_to_cluster1_distance
            )
        )
        return union_distances


# https://en.wikipedia.org/wiki/Ward%27s_method
class WardLinkage(CentroidLinkage[T], MaskedArrayRecursiveLinkage[T]):
    @override
    def distance(
        self,
        cluster1: SimpleSequence[T],
        cluster2: SimpleSequence[T],
    ) -> int | float:
        centroid_clusters_distance = super().distance(cluster1, cluster2)
        cluster1_size, cluster2_size = len(cluster1), len(cluster2)
        clusters_distance: float = (
            math.sqrt(
                2
                * (
                    (cluster1_size * cluster2_size)
                    / (cluster1_size + cluster2_size)
                )
            )
            * centroid_clusters_distance
        )
        return clusters_distance

    def joined_cluster_distances(
        self,
        old_cluster1_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster1_position: int,
        old_cluster2_distances: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        cluster2_position: int,
        old_cluster_sizes: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
    ) -> MaskedNDArray[np.integer] | MaskedNDArray[np.floating]:
        old_cluster1_size = old_cluster_sizes[cluster1_position]
        old_cluster2_size = old_cluster_sizes[cluster2_position]
        cluster1_to_cluster2_distance = old_cluster1_distances[
            cluster2_position
        ]
        cluster2_to_cluster1_distance = old_cluster2_distances[
            cluster1_position
        ]

        union_distances: MaskedNDArray[np.integer] | MaskedNDArray[np.floating]
        # np.abs() is need to fix errors caused by floating point (im)precision
        union_distances = np.sqrt(
            np.abs(
                (
                    (old_cluster1_size + old_cluster_sizes)
                    * np.square(old_cluster1_distances)
                    + (old_cluster2_size + old_cluster_sizes)
                    * np.square(old_cluster2_distances)
                    - old_cluster_sizes
                    * cluster1_to_cluster2_distance
                    * cluster2_to_cluster1_distance
                )
                / (old_cluster1_size + old_cluster2_size + old_cluster_sizes)
            )
        )
        return union_distances


class DynamicLinkage(Linkage[T]):
    def __init__(
        self,
        metric: QuasiMetric[T],
        distance_function: Callable[
            [SimpleSequence[T], SimpleSequence[T]], int | float
        ],
    ):
        self._metric = metric
        self._distance_function = distance_function

    @property
    def metric(self) -> QuasiMetric[T]:
        return self._metric

    def distance(
        self, cluster1: SimpleSequence[T], cluster2: SimpleSequence[T]
    ) -> int | float:
        return self._distance_function(cluster1, cluster2)


#
#
# ClusterDistances classes


class ClusterDistances(ABC, Generic[T]):
    @staticmethod
    def get_removed_cluster_index(
        old_cluster1_index: int,
        old_cluster2_index: int,
        cluster_union_index: int,
    ) -> int:
        old_cluster_indexes = set([old_cluster1_index, old_cluster2_index])
        assert cluster_union_index in old_cluster_indexes
        removed_cluster_indexes = list(
            old_cluster_indexes - set([cluster_union_index])
        )
        return removed_cluster_indexes[0]

    @property
    @abstractmethod
    def linkage(self) -> Linkage[T]:
        ...

    @abstractmethod
    def get_indexes_of_closest_clusters(self) -> Tuple[int, int]:
        ...

    @abstractmethod
    def update_distances_before_join(
        self,
        old_cluster1_index: int,
        old_cluster2_index: int,
        clustering: Clustering,
    ) -> None:
        ...

    @abstractmethod
    def update_distances_after_join(
        self,
        old_cluster1_index: int,
        old_cluster2_index: int,
        cluster_union_index: int,
        clustering: Clustering,
    ) -> None:
        ...


class MatrixBasedClusterDistances(ClusterDistances[T]):
    def __init__(
        self,
        linkage: Linkage[T],
        element_distances_matrix: npt.NDArray[np.integer]
        | npt.NDArray[np.floating],
    ):
        self._linkage = linkage
        self._cluster_distances_matrix: MaskedNDArray[
            np.integer
        ] | MaskedNDArray[np.floating] = np.ma.asarray(
            element_distances_matrix.copy()
        )
        # The next line is need so np.fill_diagonal() would work as expected
        self._cluster_distances_matrix.mask = np.full(
            self._cluster_distances_matrix.shape, False
        )
        self._matrix_indexes_to_cluster_indexes = {
            index: index for index in range(element_distances_matrix.shape[0])
        }
        self._cluster_indexes_to_matrix_indexes = {
            index: index for index in range(element_distances_matrix.shape[0])
        }
        self._recursive_union_distances_before_join: Optional[
            npt.NDArray[np.integer] | npt.NDArray[np.floating]
        ] = None

    @property
    def linkage(self) -> Linkage[T]:
        return self._linkage

    def get_indexes_of_closest_clusters(self) -> Tuple[int, int]:
        diagonal_mask = self._cluster_distances_matrix.mask.diagonal().copy()
        np.fill_diagonal(self._cluster_distances_matrix.mask, True)
        matrix_row, matrix_column = get_indexes_of_min_element_in_tensor(
            self._cluster_distances_matrix
        )
        np.fill_diagonal(self._cluster_distances_matrix.mask, diagonal_mask)
        cluster_index1 = self._matrix_indexes_to_cluster_indexes[matrix_row]
        cluster_index2 = self._matrix_indexes_to_cluster_indexes[matrix_column]
        min_cluster_index = min(cluster_index1, cluster_index2)
        max_cluster_index = max(cluster_index1, cluster_index2)
        return min_cluster_index, max_cluster_index

    def _get_cluster_distances_masked_array(
        self, cluster_index: int
    ) -> MaskedNDArray[np.integer] | MaskedNDArray[np.floating]:
        matrix_index = self._cluster_indexes_to_matrix_indexes[cluster_index]
        cluster_distances_masked_array: MaskedNDArray[
            np.integer
        ] | MaskedNDArray[np.floating] = self._cluster_distances_matrix[
            matrix_index, :
        ]
        return cluster_distances_masked_array

    def _get_cluster_distances_array(
        self, cluster_index: int
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        cluster_distances_masked_array = (
            self._get_cluster_distances_masked_array(cluster_index)
        )
        cluster_distances_array: npt.NDArray[np.integer] | npt.NDArray[
            np.floating
        ] = cluster_distances_masked_array.compressed()
        return cluster_distances_array

    def get_cluster_indexes_by_matrix_indexes_list(self) -> list[int]:
        cluster_indexes_by_matrix_indexes_list = [
            value
            for key, value in sorted(
                self._matrix_indexes_to_cluster_indexes.items(),
                key=operator.itemgetter(0),
            )
        ]
        return cluster_indexes_by_matrix_indexes_list

    def get_cluster_indexes_by_matrix_indexes_masked_array(
        self, old_cluster_distances_mask: npt.NDArray[np.bool_]
    ) -> MaskedNDArray[np.integer]:
        cluster_indexes_by_matrix_indexes_list = (
            self.get_cluster_indexes_by_matrix_indexes_list()
        )
        cluster_indexes_by_matrix_indexes_masked_array: MaskedNDArray[
            np.integer
        ] = np.ma.zeros((old_cluster_distances_mask.shape[0],))
        cluster_indexes_by_matrix_indexes_masked_array.mask = (
            old_cluster_distances_mask
        )
        np.place(
            cluster_indexes_by_matrix_indexes_masked_array,
            ~cluster_indexes_by_matrix_indexes_masked_array.mask,
            cluster_indexes_by_matrix_indexes_list,
        )
        return cluster_indexes_by_matrix_indexes_masked_array

    def get_union_distance_to_itself(
        self,
        old_cluster1_index: int,
        old_cluster1_distances_masked_array: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        old_matrix_index1: int,
        old_cluster2_index: int,
        old_cluster2_distances_masked_array: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        old_matrix_index2: int,
        recursive_union_distances_before_join: MaskedNDArray[np.integer]
        | MaskedNDArray[np.floating],
        clustering: Clustering,
    ) -> int | float:
        assert isinstance(self._linkage, MaskedArrayRecursiveLinkage)
        masked_tensor_recursive_linkage: MaskedArrayRecursiveLinkage = (
            self._linkage
        )

        min_matrix_index = min(old_matrix_index1, old_matrix_index2)
        max_matrix_index = max(old_matrix_index1, old_matrix_index2)
        old_cluster1_size = clustering.get_cluster_size(old_cluster1_index)
        old_cluster2_size = clustering.get_cluster_size(old_cluster2_index)
        union_distance_to_itself: int | float
        (
            union_distance_to_itself
        ) = masked_tensor_recursive_linkage.joined_cluster_distances(
            np.ma.array(
                [
                    recursive_union_distances_before_join[min_matrix_index],
                    0,
                    old_cluster1_distances_masked_array[old_matrix_index2],
                ]
            ),
            1,
            np.ma.array(
                [
                    recursive_union_distances_before_join[max_matrix_index],
                    old_cluster2_distances_masked_array[old_matrix_index1],
                    0,
                ]
            ),
            2,
            np.ma.array(
                [
                    old_cluster1_size + old_cluster2_size,
                    old_cluster1_size,
                    old_cluster2_size,
                ]
            ),
        )[
            0
        ]
        return union_distance_to_itself

    def _get_union_distances_before_join_masked_array_recursive(
        self,
        old_cluster1_index: int,
        old_cluster2_index: int,
        clustering: Clustering,
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        assert isinstance(self._linkage, MaskedArrayRecursiveLinkage)
        masked_array_recursive_linkage: MaskedArrayRecursiveLinkage = (
            self._linkage
        )

        old_matrix_index1, old_matrix_index2 = (
            self._cluster_indexes_to_matrix_indexes[old_cluster1_index],
            self._cluster_indexes_to_matrix_indexes[old_cluster2_index],
        )
        old_cluster1_distances_masked_array = (
            self._get_cluster_distances_masked_array(old_cluster1_index)
        )
        old_cluster2_distances_masked_array = (
            self._get_cluster_distances_masked_array(old_cluster2_index)
        )
        cluster_indexes_by_matrix_indexes_masked_array = (
            self.get_cluster_indexes_by_matrix_indexes_masked_array(
                old_cluster1_distances_masked_array.mask
            )
        )
        old_cluster_sizes = masked_tensor_map(
            clustering.get_cluster_size,
            cluster_indexes_by_matrix_indexes_masked_array,
        )

        recursive_union_distances_before_join_masked_array = (
            masked_array_recursive_linkage.joined_cluster_distances(
                old_cluster1_distances_masked_array,
                old_matrix_index1,
                old_cluster2_distances_masked_array,
                old_matrix_index2,
                old_cluster_sizes,
            )
        )

        min_matrix_index = min(old_matrix_index1, old_matrix_index2)
        max_matrix_index = max(old_matrix_index1, old_matrix_index2)
        recursive_union_distances_before_join_masked_array[
            min_matrix_index
        ] = self.get_union_distance_to_itself(
            old_cluster1_index,
            old_cluster1_distances_masked_array,
            old_matrix_index1,
            old_cluster2_index,
            old_cluster2_distances_masked_array,
            old_matrix_index2,
            recursive_union_distances_before_join_masked_array,
            clustering,
        )
        recursive_union_distances_before_join_masked_array[
            max_matrix_index
        ] = np.ma.masked

        recursive_union_distances_before_join: npt.NDArray[
            np.integer
        ] | npt.NDArray[
            np.floating
        ] = recursive_union_distances_before_join_masked_array.compressed()

        return recursive_union_distances_before_join

    def _get_union_distances_before_join_sequence_recursive(
        self,
        old_cluster1_index: int,
        old_cluster2_index: int,
        clustering: Clustering,
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        assert isinstance(self._linkage, SequenceRecursiveLinkage)
        sequence_recursive_linkage: SequenceRecursiveLinkage = self._linkage

        # old_matrix_index1, old_matrix_index2 = (
        #     self._cluster_indexes_to_matrix_indexes[old_cluster1_index],
        #     self._cluster_indexes_to_matrix_indexes[old_cluster2_index],
        # )

        old_cluster1_distances_array = self._get_cluster_distances_array(
            old_cluster1_index
        )
        old_cluster2_distances_array = self._get_cluster_distances_array(
            old_cluster2_index
        )
        cluster_indexes_by_matrix_indexes_list = (
            self.get_cluster_indexes_by_matrix_indexes_list()
        )
        old_cluster_sizes = [
            clustering.get_cluster_size(cluster_index)
            for cluster_index in cluster_indexes_by_matrix_indexes_list
        ]

        recursive_union_distances_before_join = (
            sequence_recursive_linkage.joined_cluster_distances(
                old_cluster1_distances_array,
                old_cluster1_index,
                old_cluster2_distances_array,
                old_cluster2_index,
                old_cluster_sizes,
            )
        )

        min_old_cluster_index = min(old_cluster1_index, old_cluster2_index)
        max_old_cluster_index = max(old_cluster1_index, old_cluster2_index)
        old_cluster1_size = clustering.get_cluster_size(old_cluster1_index)
        old_cluster2_size = clustering.get_cluster_size(old_cluster2_index)
        # min_matrix_index = min(old_matrix_index1, old_matrix_index2)
        # max_matrix_index = max(old_matrix_index1, old_matrix_index2)
        recursive_union_distances_before_join[
            min_old_cluster_index
        ] = sequence_recursive_linkage.joined_cluster_distances(
            [
                recursive_union_distances_before_join[min_old_cluster_index],
                0,
                old_cluster1_distances_array[old_cluster2_index],
            ],
            1,
            [
                recursive_union_distances_before_join[max_old_cluster_index],
                0,
                old_cluster2_distances_array[old_cluster1_index],
            ],
            2,
            [
                old_cluster1_size + old_cluster2_size,
                old_cluster1_size,
                old_cluster2_size,
            ],
        )[
            0
        ]

        recursive_union_distances_before_join.pop(max_old_cluster_index)

        return np.asarray(recursive_union_distances_before_join)

    def _get_union_distances_before_join_recursive(
        self,
        old_cluster1_index: int,
        old_cluster2_index: int,
        clustering: Clustering,
    ) -> npt.NDArray[np.integer] | npt.NDArray[np.floating]:
        if isinstance(self._linkage, MaskedArrayRecursiveLinkage):
            return (
                self._get_union_distances_before_join_masked_array_recursive(
                    old_cluster1_index,
                    old_cluster2_index,
                    clustering,
                )
            )
        elif isinstance(self._linkage, SequenceRecursiveLinkage):
            return self._get_union_distances_before_join_sequence_recursive(
                old_cluster1_index,
                old_cluster2_index,
                clustering,
            )
        else:
            supported_recursive_link_classes = (
                SequenceRecursiveLinkage,
                MaskedArrayRecursiveLinkage,
            )
            raise TypeError(
                'Class attribute self._linkage should be instance of one'
                + f'of classes from  "{supported_recursive_link_classes}".'
            )

    def update_distances_before_join(
        self,
        old_cluster1_index: int,
        old_cluster2_index: int,
        clustering: Clustering,
    ) -> None:
        if isinstance(
            self._linkage,
            (SequenceRecursiveLinkage, MaskedArrayRecursiveLinkage),
        ):
            self._recursive_union_distances_before_join = (
                self._get_union_distances_before_join_recursive(
                    old_cluster1_index,
                    old_cluster2_index,
                    clustering,
                )
            )

    def _get_union_distances_after_join_recursive(self) -> npt.NDArray:
        assert self._recursive_union_distances_before_join is not None
        union_distances = self._recursive_union_distances_before_join.copy()
        self._recursive_union_distances_before_join = None
        return union_distances

    def _get_union_distances_after_join_non_recursive(
        self,
        cluster_union_index: int,
        clustering: Clustering,
    ) -> npt.NDArray:
        cluster_union_elements = clustering.get_cluster_elements(
            cluster_union_index
        )
        clusters_indexes = clustering.get_clusters_indexes()
        union_distances = np.array(
            [
                self._linkage.distance(
                    clustering.get_cluster_elements(cluster_index),
                    cluster_union_elements,
                )
                for cluster_index in clusters_indexes
            ]
        )
        return union_distances

    def _get_union_distances_after_join(
        self,
        cluster_union_index: int,
        clustering: Clustering,
    ) -> npt.NDArray:
        if isinstance(
            self._linkage,
            (SequenceRecursiveLinkage, MaskedArrayRecursiveLinkage),
        ):
            return self._get_union_distances_after_join_recursive()
        else:
            return self._get_union_distances_after_join_non_recursive(
                cluster_union_index, clustering
            )

    def update_distances_matrix_after_join(
        self,
        old_cluster1_index: int,
        old_cluster2_index: int,
        cluster_union_index: int,
        clustering: Clustering,
    ) -> bool:
        union_distances = self._get_union_distances_after_join(
            cluster_union_index,
            clustering,
        )

        old_matrix_index1, old_matrix_index2 = (
            self._cluster_indexes_to_matrix_indexes[old_cluster1_index],
            self._cluster_indexes_to_matrix_indexes[old_cluster2_index],
        )
        min_matrix_index = min(old_matrix_index1, old_matrix_index2)
        max_matrix_index = max(old_matrix_index1, old_matrix_index2)

        # self._cluster_distances_matrix[min_matrix_index, :] = union_distances
        self._cluster_distances_matrix[
            min_matrix_index, max_matrix_index
        ] = np.ma.masked
        np.place(
            self._cluster_distances_matrix[min_matrix_index, :],
            ~self._cluster_distances_matrix.mask[min_matrix_index, :],
            union_distances,
        )

        # self._cluster_distances_matrix[:, min_matrix_index] = union_distances
        self._cluster_distances_matrix[
            max_matrix_index, min_matrix_index
        ] = np.ma.masked
        np.place(
            self._cluster_distances_matrix[:, min_matrix_index],
            ~self._cluster_distances_matrix.mask[:, min_matrix_index],
            union_distances,
        )

        old_cluster_distances_matrix_shape = (
            self._cluster_distances_matrix.shape
        )
        self._cluster_distances_matrix = delete_index_in_tensor(
            self._cluster_distances_matrix,
            index=max_matrix_index,
            mode=0,
            maximum_masked_to_data_ratio=1.0,
        )
        self._cluster_distances_matrix = delete_index_in_tensor(
            self._cluster_distances_matrix,
            index=max_matrix_index,
            mode=1,
            maximum_masked_to_data_ratio=0.7,
        )
        distances_matrix_shape_changed: bool = (
            self._cluster_distances_matrix.shape
            != old_cluster_distances_matrix_shape
        )
        return distances_matrix_shape_changed

    def update_indexes_dictionaries_after_join(
        self,
        old_cluster1_index: int,
        old_cluster2_index: int,
        cluster_union_index: int,
        distances_matrix_shape_changed: bool,
    ) -> None:
        old_matrix_index1, old_matrix_index2 = (
            self._cluster_indexes_to_matrix_indexes[old_cluster1_index],
            self._cluster_indexes_to_matrix_indexes[old_cluster2_index],
        )
        min_old_matrix_index = min(old_matrix_index1, old_matrix_index2)
        max_old_matrix_index = max(old_matrix_index1, old_matrix_index2)

        self._matrix_indexes_to_cluster_indexes[
            min_old_matrix_index
        ] = cluster_union_index
        self._matrix_indexes_to_cluster_indexes.pop(max_old_matrix_index)
        self._cluster_indexes_to_matrix_indexes.pop(old_cluster1_index)
        self._cluster_indexes_to_matrix_indexes.pop(old_cluster2_index)
        self._cluster_indexes_to_matrix_indexes[
            cluster_union_index
        ] = min_old_matrix_index

        if distances_matrix_shape_changed:
            old_matrix_indexes = sorted(
                self._matrix_indexes_to_cluster_indexes.keys()
            )
            old_matrix_indexes_to_new_matrix_indexes = {
                matrix_index: matrix_index_position
                for matrix_index_position, matrix_index in enumerate(
                    old_matrix_indexes
                )
            }
            self._matrix_indexes_to_cluster_indexes = {
                new_matrix_index: self._matrix_indexes_to_cluster_indexes[
                    old_matrix_index
                ]
                for (
                    old_matrix_index,
                    new_matrix_index,
                ) in old_matrix_indexes_to_new_matrix_indexes.items()
            }
            self._cluster_indexes_to_matrix_indexes = {
                cluster_index: old_matrix_indexes_to_new_matrix_indexes[
                    old_matrix_index
                ]
                for (
                    cluster_index,
                    old_matrix_index,
                ) in self._cluster_indexes_to_matrix_indexes.items()
            }

    def update_distances_after_join(
        self,
        old_cluster1_index: int,
        old_cluster2_index: int,
        cluster_union_index: int,
        clustering: Clustering,
    ) -> None:
        distances_matrix_shape_changed = (
            self.update_distances_matrix_after_join(
                old_cluster1_index,
                old_cluster2_index,
                cluster_union_index,
                clustering,
            )
        )

        self.update_indexes_dictionaries_after_join(
            old_cluster1_index,
            old_cluster2_index,
            cluster_union_index,
            distances_matrix_shape_changed,
        )


# pylint: disable=unnecessary-lambda
LINKAGE_FROM_METRIC: dict[str, Callable[[QuasiMetric], StaticLinkage]] = dict(
    {
        'complete': lambda metric: CompleteLinkage(metric),
        'single': lambda metric: SingleLinkage(metric),
        'average': lambda metric: AverageLinkage(metric),
        'centroid': lambda metric: CentroidLinkage(metric),
        'ward': lambda metric: WardLinkage(metric),
    }
)
# pylint: enable=unnecessary-lambda


def get_linkage_from_name(
    linkage_name: str, metric: QuasiMetric[T]
) -> StaticLinkage:
    if linkage_name not in LINKAGE_FROM_METRIC:
        raise ValueError(f'Unknown linkage "{linkage_name}".')
    return LINKAGE_FROM_METRIC[linkage_name](metric)


def get_linkage_from_distance_function(
    distance_function: Callable[
        [SimpleSequence[T], SimpleSequence[T]], int | float
    ],
    metric: QuasiMetric[T],
) -> DynamicLinkage:
    return DynamicLinkage(metric, distance_function)


def get_linkage(
    linkage_name_or_distance_function: str
    | Callable[[SimpleSequence[T], SimpleSequence[T]], int | float],
    metric: QuasiMetric[T],
) -> Linkage:
    if isinstance(linkage_name_or_distance_function, str):
        return get_linkage_from_name(linkage_name_or_distance_function, metric)
    return get_linkage_from_distance_function(
        linkage_name_or_distance_function, metric
    )


def get_cluster_distances(
    linkage_name_or_distance_function: str
    | Callable[[SimpleSequence[T], SimpleSequence[T]], int | float],
    metric: QuasiMetric[T],
    element_distances_matrix: npt.NDArray[np.integer]
    | npt.NDArray[np.floating],
) -> ClusterDistances:
    linkage = get_linkage(linkage_name_or_distance_function, metric)
    return MatrixBasedClusterDistances(linkage, element_distances_matrix)
