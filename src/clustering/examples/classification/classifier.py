from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Mapping, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from ...algorithms.util.data_structures.clustering import Clustering
from ...util.numpy_extensions import (
    convert_to_at_least_2d_tensor_inner,
    convert_to_at_least_2d_tensor_outer,
)
from .database import LabeledDatabase, TrainAndTestDatabase

if TYPE_CHECKING:
    from ...util.typing_ import SimpleSequence


T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


#
#
# Supervised Classification classes


class SupervisedClassificationEvaluation(Generic[T]):
    def _get_classes_and_labels_bitmasks(
        self,
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        classes = np.unique(
            np.concatenate((self._actual_labels, self._predicted_labels))
        )
        classes_2d_tensor = convert_to_at_least_2d_tensor_inner(classes)
        actual_labels_2d_tensor = convert_to_at_least_2d_tensor_outer(
            self._actual_labels
        )
        actual_labels_bitmask = (
            np.repeat(actual_labels_2d_tensor, classes.shape[0], axis=0)
            == classes_2d_tensor
        )
        predicted_labels_2d_tensor = convert_to_at_least_2d_tensor_outer(
            self._predicted_labels
        )
        predicted_labels_bitmask = (
            np.repeat(predicted_labels_2d_tensor, classes.shape[0], axis=0)
            == classes_2d_tensor
        )
        return classes, actual_labels_bitmask, predicted_labels_bitmask

    def _get_confusion_matrix(self) -> npt.NDArray[np.int_]:
        (
            classes,
            actual_labels_bitmask,
            predicted_labels_bitmask,
        ) = self._get_classes_and_labels_bitmasks()
        row_indexes, column_indexes = np.indices(
            (classes.shape[0], classes.shape[0])
        )
        confusion_matrix: npt.NDArray[np.int_] = np.sum(
            np.logical_and(
                predicted_labels_bitmask[row_indexes],
                actual_labels_bitmask[column_indexes],
            ),
            axis=2,
        )
        return confusion_matrix

    def _evaluate(self) -> None:
        confusion_matrix = self._get_confusion_matrix()
        accuracy = np.sum(confusion_matrix.diagonal()) / np.sum(
            confusion_matrix
        )
        precision_by_class = confusion_matrix.diagonal() / np.sum(
            confusion_matrix, axis=0
        )
        confusion_row_sums = np.sum(confusion_matrix, axis=1)
        confusion_row_sums[confusion_row_sums == 0] = 1
        recall_by_class = confusion_matrix.diagonal() / confusion_row_sums
        recall_by_class[np.isnan(recall_by_class)] = 0
        _classes, classes_counts = np.unique(
            self._actual_labels, return_counts=True
        )

        self._accuracy: float = accuracy
        self._precision: float = np.average(
            precision_by_class, weights=classes_counts
        )
        self._recall: float = np.average(
            recall_by_class, weights=classes_counts
        )
        self._f1_score = 2 / (1 / self._precision + 1 / self._recall)
        self._error_rate = 1 - self._accuracy

    def __init__(
        self,
        actual_labels: SimpleSequence[T],
        predicted_labels: SimpleSequence[T],
    ):
        self._actual_labels = np.asarray(actual_labels)
        self._predicted_labels = np.asarray(predicted_labels)
        self._evaluate()

    def _get_confusion_matrix_unoptimized(self) -> npt.NDArray:
        (
            classes,
            actual_labels_bitmask,
            predicted_labels_bitmask,
        ) = self._get_classes_and_labels_bitmasks()
        confusion_matrix = np.zeros(
            (classes.shape[0], classes.shape[0]), dtype=int
        )
        for i in range(classes.shape[0]):
            for j in range(classes.shape[0]):
                confusion_matrix[i][j] = np.sum(
                    np.logical_and(
                        actual_labels_bitmask[i], predicted_labels_bitmask[j]
                    )
                )
        return confusion_matrix

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def precision(self) -> float:
        return self._precision

    @property
    def recall(self) -> float:
        return self._recall

    @property
    def f1_score(self) -> float:
        return self._f1_score

    @property
    def error_rate(self) -> float:
        return self._error_rate


class SupervisedClassifier(ABC, Generic[U, V]):
    def __init__(self, database: TrainAndTestDatabase[U, V]):
        self._database = database
        self._train_data_set: SimpleSequence[U]
        self._train_data_labels: SimpleSequence[V]
        self._test_data_set: SimpleSequence[U]
        self._test_data_labels: SimpleSequence[V]
        (
            (self._train_data_set, self._train_data_labels),
            (self._test_data_set, self._test_data_labels),
        ) = self._database.get_train_and_test_data()

    @property
    def train_data_set(self) -> SimpleSequence[U]:
        return self._train_data_set

    @property
    def train_data_labels(self) -> SimpleSequence[V]:
        return self._train_data_labels

    @property
    def test_data_set(self) -> SimpleSequence[U]:
        return self._test_data_set

    @property
    def test_data_labels(self) -> SimpleSequence[V]:
        return self._test_data_labels

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def predict(self) -> SimpleSequence[V]:
        ...

    def test(self) -> SupervisedClassificationEvaluation:
        predicted_test_data_labels = self.predict()
        return SupervisedClassificationEvaluation(
            self._test_data_labels, predicted_test_data_labels
        )


class ClusteringSupervisedClassifier(SupervisedClassifier[U, V]):
    @property
    @abstractmethod
    def clustering(self) -> Optional[Clustering]:
        ...

    def get_cluster_classes_from_clustering(
        self, clustering: Clustering
    ) -> list[V]:
        classes_counters_by_clusters: list[collections.Counter[V]] = [
            collections.Counter() for i in range(clustering.size)
        ]

        for cluster_index, element_class in zip(
            clustering.labels, self.train_data_labels
        ):
            classes_counters_by_clusters[cluster_index][element_class] += 1
        cluster_classes = [
            classes_counter.most_common(1)[0][0]
            for classes_counter in classes_counters_by_clusters
        ]
        return cluster_classes


#
#
# Labeled Classification classes


class LabeledClassificationEvaluation:
    def _get_labels_length(self) -> int:
        assert self._actual_labels.shape[0] == self._predicted_labels.shape[0]
        return self._actual_labels.shape[0]

    # https://stats.stackexchange.com/questions/548778/how-to-compute-a-pair-confusion-matrix
    def _get_co_membership_confusion_matrix(self) -> npt.NDArray[np.int_]:
        ClassType = LabeledClassificationEvaluation
        # pylint: disable=protected-access

        actual_classes, actual_labels_inverse_indexes = np.unique(
            self._actual_labels, return_inverse=True
        )
        actual_labels_bitmask = ClassType._get_labels_bitmask(
            self._actual_labels, actual_classes
        )
        actual_labels_negated_bitmask = np.logical_not(actual_labels_bitmask)

        predicted_classes, predicted_labels_inverse_indexes = np.unique(
            self._predicted_labels, return_inverse=True
        )
        predicted_labels_bitmask = ClassType._get_labels_bitmask(
            self._predicted_labels, predicted_classes
        )
        predicted_labels_negated_bitmask = np.logical_not(
            predicted_labels_bitmask
        )

        co_membership_confusion_matrix = np.empty((2, 2), dtype=int)

        co_membership_confusion_matrix[0, 0] = (
            ClassType._get_sum_of_conjunction_of_array_indexed_tensors(
                actual_labels_bitmask,
                actual_labels_inverse_indexes,
                predicted_labels_bitmask,
                predicted_labels_inverse_indexes,
            )
            - self._labels_length
        )

        co_membership_confusion_matrix[
            0, 1
        ] = ClassType._get_sum_of_conjunction_of_array_indexed_tensors(
            actual_labels_bitmask,
            actual_labels_inverse_indexes,
            predicted_labels_negated_bitmask,
            predicted_labels_inverse_indexes,
        )

        co_membership_confusion_matrix[
            1, 0
        ] = ClassType._get_sum_of_conjunction_of_array_indexed_tensors(
            actual_labels_negated_bitmask,
            actual_labels_inverse_indexes,
            predicted_labels_bitmask,
            predicted_labels_inverse_indexes,
        )

        co_membership_confusion_matrix[
            1, 1
        ] = ClassType._get_sum_of_conjunction_of_array_indexed_tensors(
            actual_labels_negated_bitmask,
            actual_labels_inverse_indexes,
            predicted_labels_negated_bitmask,
            predicted_labels_inverse_indexes,
        )

        # pylint: enable=protected-access
        return co_membership_confusion_matrix

    # https://stackoverflow.com/questions/51294382/python-contingency-table
    def _get_frequency_cross_tabulation_matrix(self) -> npt.NDArray[np.int_]:
        actual_classes, actual_labels_inverse_indexes = np.unique(
            self._actual_labels, return_inverse=True
        )
        predicted_classes, predicted_labels_inverse_indexes = np.unique(
            self._predicted_labels, return_inverse=True
        )

        bijective_mapping_of_labels_to_vector = (
            predicted_classes.shape[0] * actual_labels_inverse_indexes
            + predicted_labels_inverse_indexes
        )
        bijective_mapping_class_counts = np.bincount(
            bijective_mapping_of_labels_to_vector
        )
        bijective_mapping_padded_class_counts = np.pad(
            bijective_mapping_class_counts,
            (
                0,
                actual_classes.shape[0] * predicted_classes.shape[0]
                - bijective_mapping_class_counts.shape[0],
            ),
        )
        frequency_cross_tabulation_matrix = (
            bijective_mapping_padded_class_counts.reshape(
                (actual_classes.shape[0], predicted_classes.shape[0])
            )
        )

        return frequency_cross_tabulation_matrix

    def _evaluate(self) -> None:
        co_membership_confusion_matrix = (
            self._get_co_membership_confusion_matrix()
        )
        # pylint: disable=invalid-name
        a = int(co_membership_confusion_matrix[0, 0])
        b = int(co_membership_confusion_matrix[0, 1])
        c = int(co_membership_confusion_matrix[1, 0])
        d = int(co_membership_confusion_matrix[1, 1])
        # pylint: enable=invalid-name

        self._precision = a / (a + c)
        self._recall = a / (a + b)
        self._f1_score = 2 / (1 / self._precision + 1 / self._recall)
        self._rand_score = (a + d) / (a + d + b + c)
        self._adjusted_rand_score = (
            2 * (a * d - b * c) / ((a + b) * (d + b) + (a + c) * (d + c))
        )

        frequency_cross_tabulation_matrix = (
            self._get_frequency_cross_tabulation_matrix()
        )

        frequency_matrix_row_sum_ratio = (
            np.sum(frequency_cross_tabulation_matrix, axis=0)
            / self._labels_length
        )
        entropy_of_actual_clustering: float = -np.dot(
            frequency_matrix_row_sum_ratio,
            np.log(frequency_matrix_row_sum_ratio),
        )

        frequency_matrix_column_sum_ratio = (
            np.sum(frequency_cross_tabulation_matrix, axis=1)
            / self._labels_length
        )
        entropy_of_predicted_clustering: float = -np.dot(
            frequency_matrix_column_sum_ratio,
            np.log(frequency_matrix_column_sum_ratio),
        )

        relative_frequency_cross_tabulation_matrix = (
            frequency_cross_tabulation_matrix / self._labels_length
        )
        frequency_matrix_nonzero_mask = (
            relative_frequency_cross_tabulation_matrix > 0
        )
        joint_entropy_of_actual_and_predicted_clustering: float = -np.sum(
            relative_frequency_cross_tabulation_matrix[
                frequency_matrix_nonzero_mask
            ]
            * np.log(
                relative_frequency_cross_tabulation_matrix[
                    frequency_matrix_nonzero_mask
                ]
            )
        )

        self._mutual_information_score = (
            entropy_of_actual_clustering
            + entropy_of_predicted_clustering
            - joint_entropy_of_actual_and_predicted_clustering
        )

        ClassType = LabeledClassificationEvaluation
        # pylint: disable=protected-access
        self._normalized_mutual_information_score = (
            self._mutual_information_score
            / ClassType._get_average_of_entropies_for_normalizing_mi(
                entropy_of_actual_clustering, entropy_of_predicted_clustering
            )
        )
        # pylint: enable=protected-access

    def __init__(
        self,
        actual_labels: SimpleSequence,
        predicted_labels: SimpleSequence,
    ):
        self._actual_labels = np.asarray(actual_labels)
        self._predicted_labels = np.asarray(predicted_labels)
        self._labels_length = self._get_labels_length()
        self._evaluate()

    @staticmethod
    def _get_average_of_entropies_for_normalizing_mi(
        entropy_of_actual_clustering: float,
        entropy_of_predicted_clustering: float,
    ) -> float:
        # Most commonly used average methods are
        # arithmetic mean, geometric mean, minimum and maximum,
        # here arithmetic mean is used.
        return (
            entropy_of_actual_clustering + entropy_of_predicted_clustering
        ) / 2

    # https://stackoverflow.com/questions/51294382/python-contingency-table
    def _get_frequency_cross_tabulation_matrix_unoptimized(
        self,
    ) -> npt.NDArray[np.int_]:
        actual_classes, actual_labels_inverse_indexes = np.unique(
            self._actual_labels, return_inverse=True
        )
        predicted_classes, predicted_labels_inverse_indexes = np.unique(
            self._predicted_labels, return_inverse=True
        )

        frequency_cross_tabulation_matrix = np.zeros(
            (actual_classes.shape[0], predicted_classes.shape[0]), dtype=int
        )
        for actual_label_inverse_index, predicted_label_inverse_index in zip(
            actual_labels_inverse_indexes, predicted_labels_inverse_indexes
        ):
            frequency_cross_tabulation_matrix[
                actual_label_inverse_index, predicted_label_inverse_index
            ] += 1

        return frequency_cross_tabulation_matrix

    @staticmethod
    def _get_sum_of_conjunction_of_array_indexed_tensors(
        tensor1: npt.NDArray[np.bool_],
        tensor1_index_array: npt.NDArray[np.int_],
        tensor2: npt.NDArray[np.bool_],
        tensor2_index_array: npt.NDArray[np.int_],
    ) -> int:
        sum_of_conjunction = 0
        for tensor1_index, tensor2_index in zip(
            tensor1_index_array, tensor2_index_array
        ):
            sum_of_conjunction += np.sum(
                np.logical_and(
                    tensor1[tensor1_index],
                    tensor2[tensor2_index],
                )
            )
        return sum_of_conjunction

    @staticmethod
    def _get_labels_bitmask(
        labels: npt.NDArray,
        classes: npt.NDArray,
    ) -> npt.NDArray[np.bool_]:
        classes_2d_tensor = convert_to_at_least_2d_tensor_inner(classes)
        labels_2d_tensor = convert_to_at_least_2d_tensor_outer(labels)
        labels_bitmask: npt.NDArray[np.bool_] = (
            np.repeat(labels_2d_tensor, classes.shape[0], axis=0)
            == classes_2d_tensor
        )
        return labels_bitmask

    # https://stats.stackexchange.com/questions/548778/how-to-compute-a-pair-confusion-matrix
    def _get_co_membership_confusion_matrix_unoptimized(
        self,
    ) -> npt.NDArray[np.int_]:
        actual_classes, actual_labels_inverse_indexes = np.unique(
            self._actual_labels, return_inverse=True
        )
        actual_labels_bitmask = (
            LabeledClassificationEvaluation._get_labels_bitmask(
                self._actual_labels, actual_classes
            )
        )

        predicted_classes, predicted_labels_inverse_indexes = np.unique(
            self._predicted_labels, return_inverse=True
        )
        predicted_labels_bitmask = (
            LabeledClassificationEvaluation._get_labels_bitmask(
                self._predicted_labels, predicted_classes
            )
        )
        co_membership_confusion_matrix = np.empty((2, 2), dtype=int)

        co_membership_confusion_matrix[0, 0] = np.sum(
            np.logical_and(
                actual_labels_bitmask[actual_labels_inverse_indexes],
                predicted_labels_bitmask[predicted_labels_inverse_indexes],
            )
        )
        co_membership_confusion_matrix[0, 0] = (
            np.sum(
                np.logical_and(
                    actual_labels_bitmask[actual_labels_inverse_indexes],
                    predicted_labels_bitmask[predicted_labels_inverse_indexes],
                )
            )
            - self._labels_length
        )

        co_membership_confusion_matrix[0, 1] = np.sum(
            np.logical_and(
                actual_labels_bitmask[actual_labels_inverse_indexes],
                np.logical_not(
                    predicted_labels_bitmask[predicted_labels_inverse_indexes]
                ),
            )
        )

        co_membership_confusion_matrix[1, 0] = np.sum(
            np.logical_and(
                np.logical_not(
                    actual_labels_bitmask[actual_labels_inverse_indexes]
                ),
                predicted_labels_bitmask[predicted_labels_inverse_indexes],
            )
        )

        co_membership_confusion_matrix[1, 1] = np.sum(
            np.logical_and(
                np.logical_not(
                    actual_labels_bitmask[actual_labels_inverse_indexes]
                ),
                np.logical_not(
                    predicted_labels_bitmask[predicted_labels_inverse_indexes]
                ),
            )
        )

        return co_membership_confusion_matrix

    @property
    def precision(self) -> float:
        return self._precision

    @property
    def recall(self) -> float:
        return self._recall

    @property
    def f1_score(self) -> float:
        return self._f1_score

    @property
    def rand_score(self) -> float:
        return self._rand_score

    @property
    def adjusted_rand_score(self) -> float:
        return self._adjusted_rand_score

    @property
    def mutual_information_score(self) -> float:
        return self._mutual_information_score

    @property
    def normalized_mutual_information_score(self) -> float:
        return self._normalized_mutual_information_score


class LabeledClassifier(ABC, Generic[U, V]):
    def __init__(self, database: LabeledDatabase[U, V]):
        self._database = database
        self._data_set: SimpleSequence[U]
        self._data_labels: SimpleSequence[V]
        (
            self._data_set,
            self._data_labels,
        ) = self._database.get_copy_of_data_and_labels()

    @property
    def data_set(self) -> SimpleSequence[U]:
        return self._data_set

    @property
    def data_labels(self) -> SimpleSequence[V]:
        return self._data_labels

    @abstractmethod
    def train(self) -> None:
        ...

    @abstractmethod
    def predict(self) -> SimpleSequence:
        ...

    def test(self) -> LabeledClassificationEvaluation:
        predicted_data_labels = self.predict()
        return LabeledClassificationEvaluation(
            self._data_labels, predicted_data_labels
        )


class ClusteringLabeledClassifier(LabeledClassifier[U, V]):
    @property
    @abstractmethod
    def clustering(self) -> Optional[Clustering]:
        ...


def get_supervised_classification_evaluations(
    classifiers: dict[str, SupervisedClassifier]
    | Mapping[str, SupervisedClassifier]
) -> dict[str, SupervisedClassificationEvaluation]:
    classification_evaluations = dict()
    for name, classifier in classifiers.items():
        classifier.train()
        classification_evaluations[name] = classifier.test()
    return classification_evaluations


def get_labeled_classification_evaluations(
    classifiers: dict[str, LabeledClassifier] | Mapping[str, LabeledClassifier]
) -> dict[str, LabeledClassificationEvaluation]:
    classification_evaluations = dict()
    for name, classifier in classifiers.items():
        classifier.train()
        classification_evaluations[name] = classifier.test()
    return classification_evaluations


def get_ar_and_nmi_text(
    classification_evaluation: LabeledClassificationEvaluation,
) -> str:
    ar_score = classification_evaluation.adjusted_rand_score
    nmi_score = classification_evaluation.normalized_mutual_information_score
    ar_and_nmi_text = f'ARI = {ar_score:.2f},    NMI = {nmi_score:.2f}'
    return ar_and_nmi_text
