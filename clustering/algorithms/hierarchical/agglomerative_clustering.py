from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

from typing_extensions import override

from ..util.data_structures.clustering import ArrayBasedClustering
from ..util.linkage import Linkage, get_cluster_distances
from ..util.metric import DistanceFunction

if TYPE_CHECKING:
    from ...util.typing_ import SimpleSequence

T = TypeVar('T')


#  P. Apolonio, Klasteriranje k-sredinama, (2018), 6–9.


class AgglomerativeClustering(ArrayBasedClustering[T]):
    def __init__(
        self,
        data: SimpleSequence[T],
        size: int,
        metric: str | DistanceFunction = 'euclidean',
        linkage: str
        | Callable[
            [SimpleSequence[T], SimpleSequence[T]], int | float
        ] = 'complete',
    ):
        super().__init__(data, size, metric)
        self._labels: list[int] = []
        self._cluster_distances = get_cluster_distances(
            linkage, self.metric, self.element_distances_matrix
        )

    @property
    def labels(self) -> list[int]:
        return self._labels

    @property
    def linkage(self) -> Linkage[T]:
        return self._cluster_distances.linkage

    @override
    def join_clusters(self, index1: int, index2: int) -> int:
        min_index, max_index = min(index1, index2), max(index1, index2)
        self._cluster_distances.update_distances_before_join(
            min_index, max_index, self
        )
        cluster_union_index = super().join_clusters(min_index, max_index)
        self._cluster_distances.update_distances_after_join(
            min_index, max_index, cluster_union_index, self
        )
        return cluster_union_index

    def analyze(self) -> None:
        while self.current_size > self.size:
            (
                cluster_index1,
                cluster_index2,
            ) = self._cluster_distances.get_indexes_of_closest_clusters()
            self.join_clusters(cluster_index1, cluster_index2)
        self.rename_subset_indexes()
        self._labels = self.subset_indexes.tolist()


def get_agglomerative_clustering(
    data: SimpleSequence[T],
    number_of_clusters: int,
    metric: str | DistanceFunction = 'euclidean',
    linkage: str
    | Callable[
        [SimpleSequence[T], SimpleSequence[T]], int | float
    ] = 'complete',
) -> AgglomerativeClustering[T]:
    """_summary_

    Arguments:
        data -- _description_
        number_of_clusters -- _description_
        metric -- _description_
    """
    agglomerative_clustering = AgglomerativeClustering(
        data, number_of_clusters, metric, linkage
    )
    agglomerative_clustering.analyze()
    return agglomerative_clustering
