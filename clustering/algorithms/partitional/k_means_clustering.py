from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import sklearn.cluster

from ..util.data_structures.clustering import ArrayBasedClustering
from ..util.metric import DistanceFunction

if TYPE_CHECKING:
    from ...util.typing_ import SimpleSequence


class KMeansClustering(ArrayBasedClustering[npt.NDArray[np.number]]):
    def __init__(
        self,
        data: SimpleSequence[npt.NDArray[np.number]],
        size: int,
        metric: str | DistanceFunction = 'euclidean',
    ):
        super().__init__(data, size, metric)
        self._labels: list[int] = []

    @property
    def labels(self) -> list[int]:
        return self._labels

    def analyze(self) -> None:
        sklearn_k_means_object = sklearn.cluster.KMeans(
            n_clusters=self.size,
            init='k-means++',
            n_init='auto',
            random_state=0,
        )
        sklearn_k_means_object.fit(self.elements)
        self._labels = sklearn_k_means_object.labels_.tolist()


def get_k_means_clustering(
    data: SimpleSequence[npt.NDArray[np.number]],
    number_of_clusters: int,
    metric: str | DistanceFunction = 'euclidean',
) -> KMeansClustering:
    k_means_clustering = KMeansClustering(data, number_of_clusters, metric)
    k_means_clustering.analyze()
    return k_means_clustering
