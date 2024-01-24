from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeVar

import numpy as np
import numpy.typing as npt

from ....algorithms.partitional.k_means_clustering import (
    KMeansClustering,
    get_k_means_clustering,
)
from ....algorithms.util.data_structures.clustering import Clustering
from ....algorithms.util.metric import DistanceFunction
from ..classifier import ClusteringLabeledClassifier
from ..database import LabeledDatabase

if TYPE_CHECKING:
    from ....util.typing_ import SimpleSequence


V = TypeVar('V')


class KMeansLabeledClassifier(
    ClusteringLabeledClassifier[npt.NDArray[np.number], V]
):
    def __init__(
        self,
        database: LabeledDatabase[npt.NDArray[np.number], V],
        metric: str | DistanceFunction = 'euclidean',
        number_of_clusters: Optional[int] = None,
    ):
        super().__init__(database)
        self._metric = metric
        self._k_means_clustering: Optional[KMeansClustering] = None
        self._cluster_classes: Optional[list[V]] = None
        self._number_of_clusters = (
            number_of_clusters
            if number_of_clusters is not None
            else len(set(self.data_labels))
        )

    @property
    def clustering(self) -> Optional[Clustering]:
        return self._k_means_clustering

    def train(self) -> None:
        self._k_means_clustering = get_k_means_clustering(
            self.data_set, self._number_of_clusters, self._metric
        )

    def predict(self) -> SimpleSequence:
        assert self._k_means_clustering is not None

        predicted_data_labels = self._k_means_clustering.labels
        return predicted_data_labels
