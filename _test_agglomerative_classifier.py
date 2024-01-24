from clustering.examples.classification.classifier import (
    SupervisedClassifier,
    get_supervised_classification_evaluations,
)

# pylint: disable-next=line-too-long
from clustering.examples.classification.clustering_classifiers.agglomerative_classifier import (
    AgglomerativeSupervisedClassifier,
)
from clustering.examples.data.mnist import MnistDatabase

#

N = 10

mnist_database = MnistDatabase(maximum_train_size=700, maximum_test_size=300)
classifiers: dict[str, SupervisedClassifier] = {
    'agglomerative classifier (complete)': AgglomerativeSupervisedClassifier(
        mnist_database,
        linkage='complete',
        number_of_clusters=N,
    ),
    'agglomerative classifier (single)': AgglomerativeSupervisedClassifier(
        mnist_database,
        linkage='single',
        number_of_clusters=N,
    ),
    'agglomerative classifier (average)': AgglomerativeSupervisedClassifier(
        mnist_database,
        linkage='average',
        number_of_clusters=N,
    ),
    'agglomerative classifier (centroid)': AgglomerativeSupervisedClassifier(
        mnist_database,
        linkage='centroid',
        number_of_clusters=N,
    ),
    'agglomerative classifier (ward)': AgglomerativeSupervisedClassifier(
        mnist_database,
        linkage='ward',
        number_of_clusters=N,
    ),
}

classifiers = dict(list(classifiers.items())[:1])

mnist_classification_evaluations = get_supervised_classification_evaluations(
    classifiers
)

for (
    classifier_name,
    classification_evaluation,
) in mnist_classification_evaluations.items():
    print(
        f'{classifier_name} - f1 score: '
        + f'{classification_evaluation.f1_score}'
    )

X = 1
