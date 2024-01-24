from typing import Mapping

from clustering.algorithms.util.data_structures.clustering import (
    ArrayBasedClustering,
)
from clustering.algorithms.util.kernel import LocalDynamicWidthGaussianKernel
from clustering.examples.classification.classifier import (
    ClusteringLabeledClassifier,
    get_ar_and_nmi_text,
    get_labeled_classification_evaluations,
)

# pylint: disable-next=line-too-long
from clustering.examples.classification.clustering_classifiers.spectral_classifier import (
    DirectedSpectralClassifier,
)
from clustering.examples.data.points_2d import (
    OverlappedBlobsDatabase,
    draw_points_2d_clustering,
)

overlapped_blobs_database = OverlappedBlobsDatabase()

# overlapped_blobs_database.draw()


# kernel_width_percentages = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
kernel_width_percentages = [
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    1,
]

# kernel_width_percentages = [
#     0.4,
#     0.45,
#     0.5,
#     0.55,
#     0.6,
#     0.65,
#     0.7,
#     0.75,
#     0.8,
#     0.85,
#     0.9,
#     0.95,
#     1,
# ]

# kernel_width_percentages = [0.1, 0.2, 0.3, 0.5, 1]
# kernel_width_percentages = [0.1, 0.2, 0.5]
# kernel_width_percentages = [0.07]

# kernel_width_percentages = [
#     0.1,
#     0.11,
#     0.12,
#     0.13,
#     0.14,
#     0.15,
#     0.16,
#     0.17,
#     0.18,
#     0.19,
#     0.2,
#     0.21,
#     0.22,
#     0.23,
#     0.24,
#     0.25,
#     0.26,
#     0.27,
#     0.28,
#     0.29,
#     0.3,
#     0.31,
#     0.32,
#     0.33,
#     0.34,
#     0.35,
#     0.36,
#     0.37,
#     0.38,
#     0.39,
#     0.4,
#     0.41,
#     0.42,
#     0.43,
#     0.44,
#     0.45,
#     0.46,
#     0.47,
#     0.48,
#     0.49,
#     0.5,
#     0.51,
#     0.52,
#     0.53,
#     0.54,
#     0.55,
#     0.56,
#     0.57,
#     0.58,
#     0.6,
#     0.6,
#     0.61,
#     0.62,
#     0.63,
#     0.64,
#     0.65,
#     0.66,
#     0.67,
#     0.68,
#     0.69,
#     0.7,
# ]

# kernel_width_percentages = [
#     0.6,
#     0.61,
#     0.62,
#     0.63,
#     0.64,
#     0.65,
#     0.66,
#     0.67,
#     0.68,
#     0.69,
#     0.7,
#     0.71,
#     0.72,
#     0.73,
#     0.74,
#     0.75,
#     0.76,
#     0.77,
#     0.78,
#     0.79,
#     0.8,
#     0.81,
#     0.82,
#     0.83,
#     0.84,
#     0.85,
#     0.86,
#     0.87,
#     0.88,
#     0.89,
#     0.9,
#     0.91,
#     0.92,
#     0.93,
#     0.94,
#     0.95,
#     0.96,
#     0.97,
#     0.98,
#     0.99,
#     1,
# ]

# kernel_width_percentages = [
#     0.77,
#     0.775,
#     0.78,
#     0.785,
#     0.79,
#     0.795,
#     0.8,
#     0.805,
#     0.81,
#     0.815,
#     0.82,
#     0.825,
#     0.83,
# ]

# kernel_width_percentages = [
#     0.22,
#     0.221,
#     0.222,
#     0.223,
#     0.224,
#     0.225,
#     0.226,
#     0.227,
#     0.228,
#     0.229,
#     0.23,
#     0.231,
#     0.232,
#     0.233,
#     0.234,
#     0.235,
#     0.236,
#     0.237,
#     0.238,
#     0.239,
#     0.24,
#     0.241,
#     0.242,
#     0.243,
#     0.244,
#     0.245,
#     0.246,
#     0.247,
#     0.248,
#     0.249,
#     0.25,
#     0.251,
#     0.252,
#     0.253,
#     0.254,
#     0.255,
#     0.256,
#     0.257,
#     0.258,
#     0.259,
#     0.26,
#     0.261,
#     0.262,
#     0.263,
#     0.264,
#     0.265,
#     0.266,
#     0.267,
#     0.268,
#     0.269,
#     0.27,
#     0.271,
#     0.272,
#     0.273,
#     0.274,
#     0.275,
#     0.276,
#     0.277,
#     0.278,
#     0.279,
#     0.28,
#     0.281,
#     0.282,
#     0.283,
#     0.284,
#     0.285,
#     0.286,
#     0.287,
#     0.288,
#     0.289,
#     0.29,
#     0.291,
#     0.292,
#     0.293,
#     0.294,
#     0.295,
#     0.296,
#     0.297,
#     0.298,
#     0.299,
#     0.3,
#     0.301,
#     0.302,
#     0.303,
#     0.304,
#     0.305,
#     0.306,
#     0.307,
#     0.308,
#     0.309,
#     0.31,
#     0.311,
#     0.312,
#     0.313,
#     0.314,
#     0.315,
#     0.316,
#     0.317,
#     0.318,
#     0.319,
#     0.32,
#     0.321,
#     0.322,
#     0.323,
#     0.324,
#     0.325,
#     0.326,
#     0.327,
#     0.328,
#     0.329,
#     0.33,
# ]

kernel_width_percentages = [
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.16,
    0.17,
    0.18,
    0.19,
    0.20,
    0.21,
    0.22,
    0.23,
    0.24,
    0.25,
    0.26,
    0.27,
    0.28,
    0.29,
    0.30,
]

kernel_width_percentages = [0.18]

# number_of_neighbors_factors = [0.1, 0.2, 0.5, 1, 2, 5, 10]
# number_of_neighbors_factors = [0.1, 0.2, 0.5]
# number_of_neighbors_factors = [1, 2, 5, 10, 20]
# number_of_neighbors_factors = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
# number_of_neighbors_factors = [1, 5, 20]
# number_of_neighbors_factors = [1, 2]
# number_of_neighbors_factors = [2, 10]
number_of_neighbors_factors = [5]

# f=0.1, p=1, r=1

neighborhood_radius_percentages = [0.3]
# neighborhood_radius_percentages = [
#     0.2,
#     0.21,
#     0.22,
#     0.23,
#     0.24,
#     0.25,
#     0.26,
#     0.27,
#     0.28,
#     0.29,
#     0.3,
#     0.31,
#     0.32,
#     0.33,
#     0.34,
#     0.35,
#     0.36,
#     0.37,
#     0.38,
#     0.39,
#     0.4,
#     0.41,
#     0.42,
#     0.43,
#     0.44,
#     0.45,
#     0.46,
#     0.47,
#     0.48,
#     0.49,
#     0.5,
# ]


# f=5,  p=0.64, r=0.1
# f=10, p=0.22, r=0.1
# f=20, p=0.20, r=0.1

# f=5,  p=0.26, r=0.2
# f=10, p=0.20, r=0.2
# f=20, p=0.27, r=0.2

# f=2,  p=0.30, r=0.3
# f=10, p=0.25, r=0.3

# f=2,  p=0.30, r=0.4
# f=10, p=0.25, r=0.4

# f=2,  p=0.30, r=0.5
# f=10, p=0.24, r=0.5

# f=2,  p=0.30, r=0.7
# f=10, p=0.24, r=0.7
# f=20, p=0.24, r=0.7

# f=2, p=0.294, r=1     (p=0.30)
# f=10, p=0.24, r=1

#
#

# f=5, p=0.23, r=0.3
# f=20, p=0.32, r=0.3

# cspell:disable-next-line
nfs_kwps_nrps = [
    (5, 0.64, 0.1),
    (10, 0.22, 0.1),
    (20, 0.20, 0.1),
    (5, 0.26, 0.2),
    (10, 0.20, 0.2),
    (20, 0.27, 0.2),
    (2, 0.30, 0.3),
    (5, 0.23, 0.3),
    (10, 0.25, 0.3),
    (20, 0.32, 0.3),
    (2, 0.30, 0.4),
    (10, 0.25, 0.4),
    (2, 0.30, 0.5),
    (10, 0.24, 0.5),
    (2, 0.30, 0.7),
    (10, 0.24, 0.7),
    (20, 0.24, 0.7),
    (2, 0.30, 1),
    (10, 0.24, 1),
]


classifiers: Mapping[str, ClusteringLabeledClassifier] = {
    (
        'Directed spectral clustering on overlapped blobs '
        + f'- f={factor},  σ={kernel_width},  r={neighborhood_radius}'
    ): DirectedSpectralClassifier(
        overlapped_blobs_database,
        kernel=LocalDynamicWidthGaussianKernel(
            kernel_width, factor, neighborhood_radius
        ),
    )
    for factor in number_of_neighbors_factors
    for neighborhood_radius in neighborhood_radius_percentages
    for kernel_width in kernel_width_percentages
    # cspell:disable-next-line
    # for factor, kernel_width, neighborhood_radius in nfs_kwps_nrps
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
    print()

# directed_spectral_clustering = classifiers[
#     'Directed spectral clustering on overlapped blobs'
# ].clustering
# assert isinstance(directed_spectral_clustering, ArrayBasedClustering)

classifier_name = next(iter(classifiers))
directed_spectral_clustering = classifiers[classifier_name].clustering
assert isinstance(directed_spectral_clustering, ArrayBasedClustering)

evaluation_text = get_ar_and_nmi_text(
    classification_evaluations[classifier_name]
)
parameters_text = classifier_name.split('-')[1].strip()

draw_points_2d_clustering(
    directed_spectral_clustering, evaluation_text, parameters_text
)

X = 1
