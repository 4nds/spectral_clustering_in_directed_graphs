from typing import Mapping

from clustering.algorithms.util.data_structures.clustering import (
    ArrayBasedClustering,
)
from clustering.algorithms.util.kernel import LocalConstantWidthGaussianKernel
from clustering.examples.classification.classifier import (
    ClusteringLabeledClassifier,
    get_ar_and_nmi_text,
    get_labeled_classification_evaluations,
)

# pylint: disable-next=line-too-long
from clustering.examples.classification.clustering_classifiers.spectral_classifier import (
    UndirectedSpectralClassifier,
)

# pylint: disable=unused-import
from clustering.examples.data.points_2d import (
    OverlappedBlobsDatabase,
    draw_points_2d_clustering,
)

# pylint: enable=unused-import

overlapped_blobs_database = OverlappedBlobsDatabase()

# overlapped_blobs_database.draw()

# classifiers: Mapping[str, ClusteringLabeledClassifier] = {
#     (
#         'Undirected spectral clustering on overlapped blobs'
#     ): UndirectedSpectralClassifier(overlapped_blobs_database)
# }

# kernel_width_percentages = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
# kernel_width_percentages = [0.01, 0.02, 0.05, 0.1, 0.2]
# kernel_width_percentages = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# kernel_width_percentages = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# kernel_width_percentages = [0.03, 0.035, 0.04, 0.045, 0.05]
kernel_width_percentages = [0.04]


# neighborhood_radius_percentages = [1]
# neighborhood_radius_percentages = [0.8]
# neighborhood_radius_percentages = [0.6]
# neighborhood_radius_percentages = [0.5]
# neighborhood_radius_percentages = [0.4]
neighborhood_radius_percentages = [0.3]
# neighborhood_radius_percentages = [0.2]
# neighborhood_radius_percentages = [0.2]
# neighborhood_radius_percentages = [0.1]
# neighborhood_radius_percentages = [0.05]
# neighborhood_radius_percentages = [0.03]
# neighborhood_radius_percentages = [0.02]
# neighborhood_radius_percentages = [0.01]
# neighborhood_radius_percentages = [
#     0.15,
#     0.2,
#     0.3,
#     0.4,
#     0.5,
#     0.6,
#     0.7,
#     0.8,
#     0.9,
#     1,
# ]

# p=0.04 r=0.2
# p=0.04 r=0.3
# p=0.04 r=0.4
# p=0.04 r=0.5
# p=0.04 r=0.6
# p=0.04 r=0.8
# p=0.04 r=1

# cspell:disable-next-line
kwps_nrps = [
    (0.04, 0.5),
    (0.25, 0.5),
    (0.31, 0.5),
]


classifiers: Mapping[str, ClusteringLabeledClassifier] = {
    (
        'Undirected spectral clustering on overlapped blobs '
        + f'- σ={kernel_width}  r={neighborhood_radius}'
    ): UndirectedSpectralClassifier(
        overlapped_blobs_database,
        kernel=LocalConstantWidthGaussianKernel(
            kernel_width, neighborhood_radius
        ),
    )
    for neighborhood_radius in neighborhood_radius_percentages
    for kernel_width in kernel_width_percentages
    # cspell:disable-next-line
    # for kernel_width, neighborhood_radius in kwps_nrps
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

# undirected_spectral_clustering = classifiers[
#     'Undirected spectral clustering on overlapped blobs'
# ].clustering
# assert isinstance(undirected_spectral_clustering, ArrayBasedClustering)

classifier_name = next(iter(classifiers))
undirected_spectral_clustering = classifiers[classifier_name].clustering
assert isinstance(undirected_spectral_clustering, ArrayBasedClustering)

evaluation_text = get_ar_and_nmi_text(
    classification_evaluations[classifier_name]
)
parameters_text = classifier_name.split('-')[1].strip()

draw_points_2d_clustering(
    undirected_spectral_clustering, evaluation_text, parameters_text
)

X = 1
