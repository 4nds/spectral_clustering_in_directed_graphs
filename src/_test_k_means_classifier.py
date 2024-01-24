from clustering.examples.classification.classifier import (
    LabeledClassifier,
    get_labeled_classification_evaluations,
)

# pylint: disable-next=line-too-long
from clustering.examples.classification.clustering_classifiers.k_means_classifier import (
    KMeansLabeledClassifier,
)
from clustering.examples.data.points_2d import BlobsDatabase

blobs_database = BlobsDatabase(4)

classifiers: dict[str, LabeledClassifier] = {
    'k_means classifier': KMeansLabeledClassifier(blobs_database)
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


X = 1
