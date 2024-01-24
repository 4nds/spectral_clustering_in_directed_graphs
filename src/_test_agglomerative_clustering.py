from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import scipy.cluster.hierarchy
import scipy.spatial.distance
import sklearn.cluster

from clustering.algorithms.hierarchical.agglomerative_clustering import (
    get_agglomerative_clustering,
)
from clustering.algorithms.util.linkage import get_linkage
from clustering.algorithms.util.metric import get_tensor_quasi_metric
from clustering.examples.data.mnist import MnistDatabase

if TYPE_CHECKING:
    from clustering.util.typing_ import SimpleSequence


def sort_labels(
    labels: SimpleSequence[int] | SimpleSequence[np.integer],
) -> list[int]:
    old_to_new_labels: dict[int, int] = dict()
    new_labels = []
    for label in labels:
        new_labels.append(
            old_to_new_labels.setdefault(int(label), len(old_to_new_labels))
        )
    return new_labels


def get_cluster_label(
    cluster_index: int,
    scipy_linkage_matrix_: npt.NDArray,
    scipy_clustering_labels_: npt.NDArray[np.int_],
) -> int:
    while cluster_index > scipy_linkage_matrix_.shape[0]:
        cluster_index -= scipy_linkage_matrix_.shape[0] + 1
        cluster_index = int(scipy_linkage_matrix_[cluster_index, 0])
    return cast(int, scipy_clustering_labels_[cluster_index])


def get_labels_from_scipy_linkage_matrix(
    scipy_linkage_matrix_: npt.NDArray, number_of_clusters: int
) -> npt.NDArray[np.int_]:
    scipy_clustering_labels_: npt.NDArray[np.int_] = np.array([], dtype=int)
    number_of_labels = 0
    i = 0
    while number_of_labels < number_of_clusters:
        scipy_clustering_labels_ = scipy.cluster.hierarchy.fcluster(
            scipy_linkage_matrix_, number_of_clusters + i, criterion='maxclust'
        )
        number_of_labels = len(set(scipy_clustering_labels_))
        i += 1
    for i in range(number_of_labels, number_of_clusters, -1):
        cluster1_index, cluster2_index = scipy_linkage_matrix_[
            -i + 1, :2
        ].astype(int)
        cluster1_label = get_cluster_label(
            cluster1_index, scipy_linkage_matrix_, scipy_clustering_labels_
        )
        cluster2_label = get_cluster_label(
            cluster2_index, scipy_linkage_matrix_, scipy_clustering_labels_
        )
        min_cluster_label = min(cluster1_label, cluster2_label)
        max_cluster_label = max(cluster1_label, cluster2_label)
        scipy_clustering_labels_[
            scipy_clustering_labels_ == max_cluster_label
        ] = min_cluster_label
    return scipy_clustering_labels_


np.seterr('raise')

euclidean_metric = get_tensor_quasi_metric('euclidean')
complete_linkage = get_linkage('complete', euclidean_metric)

# mnist_database = MnistDatabase(maximum_train_size=7, maximum_test_size=7)
# mnist_database = MnistDatabase(maximum_train_size=15, maximum_test_size=8)
# mnist_database = MnistDatabase(maximum_train_size=20, maximum_test_size=10)
# mnist_database = MnistDatabase(maximum_train_size=30, maximum_test_size=16)
# mnist_database = MnistDatabase(maximum_train_size=70, maximum_test_size=30)
mnist_database = MnistDatabase(maximum_train_size=700, maximum_test_size=300)
# mnist_database = MnistDatabase(
#     maximum_train_size=7000,
#     maximum_test_size=3000,
# )

(
    (train_data_set, train_data_labels),
    (test_data_set, test_data_labels),
) = mnist_database.get_train_and_test_data()

train_data_set = np.asarray(train_data_set)
reshaped_train_data_set = train_data_set.reshape(train_data_set.shape[0], -1)

sorted_train_data_labels = sort_labels(train_data_labels)

# LINKAGE_NAME = 'complete'
# LINKAGE_NAME = 'single'
# LINKAGE_NAME = 'average'
# LINKAGE_NAME = 'centroid'  # provjeri na velikom uzorku
LINKAGE_NAME = 'ward'


X = 1

print('Clustering started\n')

inter_t1 = time.time()
agglomerative_clustering = get_agglomerative_clustering(
    train_data_set, 10, linkage=LINKAGE_NAME
)
inter_t2 = time.time()

agglomerative_clustering_labels = sort_labels(agglomerative_clustering.labels)


scipy_t1 = time.time()
scipy_linkage_matrix = scipy.cluster.hierarchy.linkage(
    reshaped_train_data_set, method=LINKAGE_NAME
)

# scipy_clustering_labels_not_sorted = scipy.cluster.hierarchy.fcluster(
#     scipy_linkage_matrix, 10, criterion='maxclust'
# )
scipy_clustering_labels_not_sorted = get_labels_from_scipy_linkage_matrix(
    scipy_linkage_matrix, 10
)
scipy_t2 = time.time()

scipy_clustering_labels = sort_labels(scipy_clustering_labels_not_sorted)


if LINKAGE_NAME in ('complete', 'single', 'average', 'ward'):
    sklearn_t1 = time.time()
    sklearn_agglomerative_clustering = sklearn.cluster.AgglomerativeClustering(
        n_clusters=10, linkage=LINKAGE_NAME
    ).fit(reshaped_train_data_set)
    sklearn_t2 = time.time()

    sklearn_clustering_labels = sort_labels(
        sklearn_agglomerative_clustering.labels_
    )
else:
    sklearn_clustering_labels = []
    sklearn_t1, sklearn_t2 = 0, 0


print(sorted_train_data_labels[:30])
print()
print(agglomerative_clustering.labels[:30])
print(agglomerative_clustering_labels[:30])
print(inter_t2 - inter_t1)
print()
if LINKAGE_NAME in ('complete', 'single', 'average', 'ward'):
    print(sklearn_clustering_labels[:30])
    print(
        np.all(
            np.array(agglomerative_clustering_labels)
            == np.array(sklearn_clustering_labels)
        )
    )
    print(sklearn_t2 - sklearn_t1)
print()
print(scipy_clustering_labels[:30])
print(
    np.all(
        np.array(agglomerative_clustering_labels)
        == np.array(scipy_clustering_labels)
    )
)
print(scipy_t2 - scipy_t1)
print()
print()
# with np.printoptions(precision=3, suppress=True):
#     print(scipy_linkage_matrix)

# cluster1 = [reshaped_train_data_set[3]]
# cluster2 = [reshaped_train_data_set[23]]
# element_distances_matrix = euclidean_metric.pairwise_distances(
#     cluster1, cluster2
# )


X = 1
