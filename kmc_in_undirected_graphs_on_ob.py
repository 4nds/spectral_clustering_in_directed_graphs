from typing import Mapping

from clustering.algorithms.util.data_structures.clustering import (
    ArrayBasedClustering,
)
from clustering.examples.classification.classifier import (
    ClusteringLabeledClassifier,
    get_ar_and_nmi_text,
    get_labeled_classification_evaluations,
)

# pylint: disable-next=line-too-long
from clustering.examples.classification.clustering_classifiers.k_means_classifier import (
    KMeansLabeledClassifier,
)
from clustering.examples.data.points_2d import (
    OverlappedBlobsDatabase,
    draw_points_2d_clustering,
)

overlapped_blobs_database = OverlappedBlobsDatabase()

# blob_and_line_database.draw()

classifiers: Mapping[str, ClusteringLabeledClassifier] = {
    'K_means clustering on overlapped blobs': KMeansLabeledClassifier(
        overlapped_blobs_database
    )
}

classification_evaluations = get_labeled_classification_evaluations(
    classifiers
)

for (
    classifier_name,
    classification_evaluation,
) in classification_evaluations.items():
    print(
        f'{classifier_name} - adjusted rand: '
        + f'{classification_evaluation.adjusted_rand_score}'
    )
    print(
        f'{classifier_name} - normalized mutual information: '
        + f'{classification_evaluation.normalized_mutual_information_score}'
    )

k_means_clustering = classifiers[
    'K_means clustering on overlapped blobs'
].clustering
assert isinstance(k_means_clustering, ArrayBasedClustering)

evaluation_text = get_ar_and_nmi_text(
    classification_evaluations['K_means clustering on overlapped blobs']
)

draw_points_2d_clustering(k_means_clustering, evaluation_text)
