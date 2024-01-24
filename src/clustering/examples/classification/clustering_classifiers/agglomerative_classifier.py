from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

from ....algorithms.hierarchical.agglomerative_clustering import (
    AgglomerativeClustering,
    get_agglomerative_clustering,
)
from ....algorithms.util.metric import DistanceFunction
from ..classifier import ClusteringSupervisedClassifier
from ..database import TrainAndTestDatabase

if TYPE_CHECKING:
    from ....util.typing_ import SimpleSequence

U = TypeVar('U')
V = TypeVar('V')


class AgglomerativeSupervisedClassifier(ClusteringSupervisedClassifier[U, V]):
    def __init__(
        self,
        database: TrainAndTestDatabase[U, V],
        metric: str | DistanceFunction = 'euclidean',
        linkage: str
        | Callable[
            [SimpleSequence[U], SimpleSequence[U]], int | float
        ] = 'complete',
        number_of_clusters: Optional[int] = None,
    ):
        super().__init__(database)
        self._metric = metric
        self._linkage = linkage
        self._agglomerative_clustering: Optional[
            AgglomerativeClustering
        ] = None
        self._cluster_classes: Optional[list[V]] = None
        self._number_of_clusters = (
            number_of_clusters
            if number_of_clusters is not None
            else len(set(self.train_data_labels))
        )

    def train(self) -> None:
        self._agglomerative_clustering = get_agglomerative_clustering(
            self.train_data_set,
            self._number_of_clusters,
            self._metric,
            self._linkage,
        )
        self._cluster_classes = self.get_cluster_classes_from_clustering(
            self._agglomerative_clustering
        )

    def predict(self) -> SimpleSequence[V]:
        assert self._agglomerative_clustering is not None
        assert self._cluster_classes is not None

        clusters = [
            self._agglomerative_clustering.get_cluster_elements(cluster_index)
            for cluster_index in range(self._agglomerative_clustering.size)
        ]
        predicted_test_data_labels = []
        for element in self.test_data_set:
            distances_to_clusters = (
                self._agglomerative_clustering.linkage.distance(
                    cluster, [element]
                )
                for cluster in clusters
            )
            closest_cluster, _closest_distance = min(
                enumerate(distances_to_clusters), key=operator.itemgetter(1)
            )
            predicted_test_data_labels.append(
                self._cluster_classes[closest_cluster]
            )
        return predicted_test_data_labels
