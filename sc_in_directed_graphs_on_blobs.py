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
    BlobsDatabase,
    draw_points_2d_clustering,
)

blobs_database = BlobsDatabase(4)

# blobs_database.draw()

kernel_width_percentages = [0.3]

number_of_neighbors_factors = [20]

neighborhood_radius_percentages = [0.2]

classifiers: Mapping[str, ClusteringLabeledClassifier] = {
    (
        'Directed spectral clustering on blobs '
        + f'- f={factor},  σ={kernel_width},  r={neighborhood_radius}'
    ): DirectedSpectralClassifier(
        blobs_database,
        kernel=LocalDynamicWidthGaussianKernel(
            kernel_width, factor, neighborhood_radius
        ),
    )
    for factor in number_of_neighbors_factors
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
