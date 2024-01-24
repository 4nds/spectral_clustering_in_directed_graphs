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

# pylint: disable=unused-import
from clustering.examples.data.points_2d import (
    BlobAndLineDatabase,
    draw_points_2d_clustering,
)

# pylint: enable=unused-import

blob_and_line_database = BlobAndLineDatabase()

# blob_and_line_database.draw()


# classifiers: Mapping[str, ClusteringLabeledClassifier] = {
#     (
#         'Directed spectral clustering on blob and line'
#     ): DirectedSpectralClassifier(blob_and_line_database)
# }


# kernel_width_percentages = [0.1, 0.2, 0.3, 0.5, 1]
# kernel_width_percentages = [0.3, 0.35, 0.4, 0.45, 0.5]
# kernel_width_percentages = [0.3]
# kernel_width_percentages = [0.05, 0.1, 0.2]
# kernel_width_percentages = [0.1, 0.2, 0.5]
kernel_width_percentages = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
# kernel_width_percentages = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# kernel_width_percentages = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# kernel_width_percentages = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
# kernel_width_percentages = [0.3, 0.35, 0.4, 0.45, 0.5]
# kernel_width_percentages = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
# kernel_width_percentages = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# kernel_width_percentages = [0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.77, 0.8]
# kernel_width_percentages = [0.57, 0.575, 0.58, 0.585, 0.59]
# kernel_width_percentages = [0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53]
# kernel_width_percentages = [0.4, 0.41, 0.42, 0.43, 0.44, 0.45]
# kernel_width_percentages = [0.35, 0.37, 0.38, 0.39, 0.4, 0.42, 0.44, 0.45]
# kernel_width_percentages = [0.2, 0.22, 0.23, 0.24, 0.25, 0.26, 0.28, 0.3]
# kernel_width_percentages = [0.06, 0.065, 0.07, 0.075, 0.08, 0.09]
# kernel_width_percentages = [0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.075]
# kernel_width_percentages = [0.27, 0.275, 0.28, 0.285, 0.29]
# kernel_width_percentages = [0.13, 0.135, 0.14, 0.145, 0.15]
# kernel_width_percentages = [0.09, 0.095, 0.1, 0.105, 0.11]
# kernel_width_percentages = [0.23, 0.235, 0.24, 0.245, 0.25]
# kernel_width_percentages = [0.002, 0.005, 0.007, 0.01, 0.02]
# kernel_width_percentages = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
kernel_width_percentages = [0.007]

# number_of_neighbors_factors = [0.1, 0.2, 0.5, 1, 2, 5, 10]
# number_of_neighbors_factors = [5, 10, 15, 20, 30]
# number_of_neighbors_factors = [2, 3, 4, 5, 6, 8, 10]
# number_of_neighbors_factors = [3, 3.3, 3.6, 4, 4.3, 4.6, 5]
# number_of_neighbors_factors = [4]
# number_of_neighbors_factors = [2, 5, 10]
number_of_neighbors_factors = [20]


# neighborhood_radius_percentages = [0.02, 0.03, 0.05]
neighborhood_radius_percentages = [0.03]


# f= 2, p=0.72, r=0.03
# f= 4, p=0.49, r=0.03
# f= 5, p=0.42, r=0.03
# f=10, p=0.24, r=0.03
# f=20, p=0.07, r=0.03

# f= 5, p=0.38, r=0.1
# f=10, p=0.28, r=0.1
# f=10, p=0.58, r=0.1
# f=20, p=0.07, r=0.1

# cspell:disable
# nfs_kwps_nrps = [
#     (2, 0.72, 0.03),
#     (4, 0.49, 0.03),
#     (5, 0.42, 0.03),
#     (10, 0.24, 0.03),
#     (20, 0.07, 0.03),
#     (20, 0.0676, 0.03),
# ]

# nfs_kwps_nrps = [
#     (10, 0.1, 0.1),
#     (10, 0.14, 0.1),
#     (10, 0.28, 0.1),
#     (10, 0.58, 0.1),
# ]
# cspell:enable


classifiers: Mapping[str, ClusteringLabeledClassifier] = {
    (
        'Directed spectral clustering on blob and line '
        + f'- f={factor},  σ={kernel_width},  r={neighborhood_radius}'
    ): DirectedSpectralClassifier(
        blob_and_line_database,
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
#     'Directed spectral clustering on blob and line - f=10, p=0.1'
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
