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
    BlobAndLineDatabase,
    draw_points_2d_clustering,
)

# pylint: enable=unused-import

blob_and_line_database = BlobAndLineDatabase()

# blob_and_line_database.draw()

# classifiers: Mapping[str, ClusteringLabeledClassifier] = {
#     (
#         'Undirected spectral clustering on blob and line'
#     ): UndirectedSpectralClassifier(blob_and_line_database)
# }


# kernel_width_percentages = [0.05, 0.1, 0.2, 0.3, 0.5, 1]
# kernel_width_percentages = [0.01, 0.02, 0.05, 0.1, 0.2]
# kernel_width_percentages = [0.001, 0.002, 0.005, 0.007, 0.01]
# kernel_width_percentages = [0.001, 0.002, 0.005, 0.01, 0.02]
# kernel_width_percentages = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# kernel_width_percentages = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
# kernel_width_percentages = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.02]
# kernel_width_percentages = [0.006, 0.007, 0.008, 0.009, 0.01]
# kernel_width_percentages = [0.006, 0.007, 0.008, 0.009, 0.01]
# kernel_width_percentages = [0.006, 0.0065, 0.0066, 0.0067, 0.0068, 0.007]
# kernel_width_percentages = [0.0066, 0.00661, 0.00662, 0.00663, 0.00665]
# kernel_width_percentages = [0.006622, 0.006623, 0.006624, 0.006625, 0.006626]
# kernel_width_percentages = [0.006624, 0.0066248, 0.0066249, 0.006625]
# kernel_width_percentages = [0.0066248, 0.00662481, 0.00662482, 0.00662485]
# kernel_width_percentages = [0.006624804, 0.006624805, 0.006624806]
# kernel_width_percentages = [0.006625, 0.007, 0.01, 0.02, 0.05]
# kernel_width_percentages = [0.02, 0.1]
kernel_width_percentages = [0.05]

# neighborhood_radius_percentages = [0.05, 0.1, 0.2, 0.5]
# neighborhood_radius_percentages = [0.02, 0.0286, 0.03]
# neighborhood_radius_percentages = [0.02, 0.03, 0.05, 1]
# neighborhood_radius_percentages = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3]
neighborhood_radius_percentages = [0.03]


# p=0.05, r=0.03
# p=0.006625 r=0.03

classifiers: Mapping[str, ClusteringLabeledClassifier] = {
    (
        'Undirected spectral clustering on blob and line '
        + f'- σ={kernel_width}  r={neighborhood_radius}'
    ): UndirectedSpectralClassifier(
        blob_and_line_database,
        kernel=LocalConstantWidthGaussianKernel(
            kernel_width, neighborhood_radius
        ),
    )
    for neighborhood_radius in neighborhood_radius_percentages
    for kernel_width in kernel_width_percentages
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

# undirected_spectral_clustering = classifiers[
#     'Undirected spectral clustering on blob and line'
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
