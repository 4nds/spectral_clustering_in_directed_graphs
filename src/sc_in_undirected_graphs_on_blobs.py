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
from clustering.examples.data.points_2d import (
    BlobsDatabase,
    draw_points_2d_clustering,
)

blobs_database = BlobsDatabase(4)

# blobs_database.draw()

kernel_width_percentages = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
kernel_width_percentages = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
kernel_width_percentages = [0.3]

neighborhood_radius_percentages = [0.1, 0.2, 0.3, 0.5, 1]
neighborhood_radius_percentages = [0.2]

classifiers: Mapping[str, ClusteringLabeledClassifier] = {
    (
        'Undirected spectral clustering on blobs '
        + f'- σ={kernel_width}  r={neighborhood_radius}'
    ): UndirectedSpectralClassifier(
        blobs_database,
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
