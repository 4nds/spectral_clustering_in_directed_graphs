from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TypeVar

import numpy as np
import numpy.typing as npt

from ....algorithms.partitional.spectral_clustering import (
    DirectedSpectralClustering,
    UndirectedSpectralClustering,
    get_directed_spectral_clustering,
    get_undirected_spectral_clustering,
)
from ....algorithms.util.data_structures.clustering import Clustering
from ....algorithms.util.kernel import (
    CONSTANT_WIDTH_GAUSSIAN_KERNEL,
    DYNAMIC_WIDTH_GAUSSIAN_KERNEL,
    Kernel,
)
from ....algorithms.util.metric import DistanceFunction
from ..classifier import ClusteringLabeledClassifier
from ..database import LabeledDatabase

if TYPE_CHECKING:
    from ....util.typing_ import SimpleSequence


V = TypeVar('V')


class UndirectedSpectralClassifier(
    ClusteringLabeledClassifier[npt.NDArray[np.number], V]
):
    def __init__(
        self,
        database: LabeledDatabase[npt.NDArray[np.number], V],
        metric: str | DistanceFunction = 'euclidean',
        kernel: Kernel = CONSTANT_WIDTH_GAUSSIAN_KERNEL,
        number_of_clusters: Optional[int] = None,
    ):
        super().__init__(database)
        self._metric = metric
        self._kernel = kernel
        self._undirected_spectral_clustering: Optional[
            UndirectedSpectralClustering
        ] = None
        self._cluster_classes: Optional[list[V]] = None
        self._number_of_clusters = (
            number_of_clusters
            if number_of_clusters is not None
            else len(set(self.data_labels))
        )

    @property
    def clustering(self) -> Optional[Clustering]:
        return self._undirected_spectral_clustering

    def train(self) -> None:
        self._undirected_spectral_clustering = (
            get_undirected_spectral_clustering(
                self.data_set,
                self._number_of_clusters,
                self._metric,
                self._kernel,
            )
        )

    def predict(self) -> SimpleSequence:
        assert self._undirected_spectral_clustering is not None

        predicted_data_labels = self._undirected_spectral_clustering.labels
        return predicted_data_labels


class DirectedSpectralClassifier(
    ClusteringLabeledClassifier[npt.NDArray[np.number], V]
):
    def __init__(
        self,
        database: LabeledDatabase[npt.NDArray[np.number], V],
        metric: str | DistanceFunction = 'euclidean',
        kernel: Kernel = DYNAMIC_WIDTH_GAUSSIAN_KERNEL,
        number_of_clusters: Optional[int] = None,
    ):
        super().__init__(database)
        self._metric = metric
        self._kernel = kernel
        self._directed_spectral_clustering: Optional[
            DirectedSpectralClustering
        ] = None
        self._cluster_classes: Optional[list[V]] = None
        self._number_of_clusters = (
            number_of_clusters
            if number_of_clusters is not None
            else len(set(self.data_labels))
        )

    @property
    def clustering(self) -> Optional[Clustering]:
        return self._directed_spectral_clustering

    def train(self) -> None:
        self._directed_spectral_clustering = get_directed_spectral_clustering(
            self.data_set,
            self._number_of_clusters,
            self._metric,
            self._kernel,
        )

    def predict(self) -> SimpleSequence:
        assert self._directed_spectral_clustering is not None

        predicted_data_labels = self._directed_spectral_clustering.labels
        return predicted_data_labels
