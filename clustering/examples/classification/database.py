from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Optional, Tuple, TypeVar

from .data_extractor import (
    DEFAULT_LABEL_EXTRACTOR,
    FeatureExtractor,
    LabelExtractor,
)
from .util.convert import as_list

if TYPE_CHECKING:
    from ...util.typing_ import SimpleSequence


U = TypeVar('U')
V = TypeVar('V')
W = TypeVar('W')
X = TypeVar('X')


#
#
# Database classes


class Database(ABC, Generic[U]):
    def __init__(
        self,
        maximum_size: Optional[int] = None,
    ):
        self._maximum_size = maximum_size

    @abstractmethod
    def get_data(self) -> SimpleSequence[U]:
        ...

    @abstractmethod
    def get_copy_of_data(self) -> SimpleSequence[U]:
        ...


class PreparedDatabase(Database[U]):
    ...


class PreprocessedDatabase(Database[U]):
    def __init__(
        self,
        processed_data_set: SimpleSequence[U],
        maximum_size: Optional[int] = None,
    ):
        super().__init__(maximum_size)
        self._processed_data_set = processed_data_set

    def get_data(self) -> SimpleSequence[U]:
        return self._processed_data_set


class RawDatabase(PreprocessedDatabase[U]):
    def __init__(
        self,
        raw_data_set: SimpleSequence,
        feature_extractor: FeatureExtractor,
        maximum_size: Optional[int] = None,
    ):
        self._raw_data_set = as_list(raw_data_set)
        processed_data_set: list[U] = list(
            feature_extractor.extract(self._raw_data_set)
        )

        super().__init__(processed_data_set, maximum_size)
        self._feature_extractor = feature_extractor

    def get_copy_of_data(self) -> SimpleSequence[U]:
        return list(self._processed_data_set)


#
#
# LabeledDatabase classes


class LabeledDatabase(Database[U], Generic[U, V]):
    @abstractmethod
    def get_data_and_labels(
        self,
    ) -> Tuple[SimpleSequence[U], SimpleSequence[V]]:
        ...

    @abstractmethod
    def get_copy_of_data_and_labels(
        self,
    ) -> Tuple[SimpleSequence[U], SimpleSequence[V]]:
        ...


class PreparedLabeledDatabase(LabeledDatabase[U, V]):
    ...


class PreprocessedLabeledDatabase(LabeledDatabase[U, V]):
    def __init__(
        self,
        processed_data_set: SimpleSequence[U],
        processed_data_labels: SimpleSequence[V],
        maximum_size: Optional[int] = None,
    ):
        super().__init__(maximum_size)
        self._processed_data_set = processed_data_set[:maximum_size]
        self._processed_data_labels = processed_data_labels[:maximum_size]

    def get_data(self) -> SimpleSequence[U]:
        return self._processed_data_set

    def get_data_and_labels(
        self,
    ) -> Tuple[SimpleSequence[U], SimpleSequence[V]]:
        return self._processed_data_set, self._processed_data_labels


class RawLabeledDatabase(PreprocessedLabeledDatabase[U, V]):
    def __init__(
        self,
        raw_data_set: SimpleSequence,
        raw_data_labels: SimpleSequence,
        feature_extractor: FeatureExtractor,
        label_extractor: LabelExtractor = DEFAULT_LABEL_EXTRACTOR,
        maximum_size: Optional[int] = None,
    ):
        self._raw_data_set = as_list(raw_data_set)
        processed_data_set: list[U] = list(
            feature_extractor.extract(self._raw_data_set)
        )
        self._raw_data_labels = as_list(raw_data_labels)
        processed_data_labels: list[V] = list(
            label_extractor.extract(self._raw_data_labels)
        )

        super().__init__(
            processed_data_set, processed_data_labels, maximum_size
        )
        self._feature_extractor = feature_extractor
        self._label_extractor = label_extractor

    def get_copy_of_data(self) -> SimpleSequence[U]:
        return list(self._processed_data_set)

    def get_copy_of_data_and_labels(
        self,
    ) -> Tuple[SimpleSequence[U], SimpleSequence[V]]:
        return list(self._processed_data_set), list(
            self._processed_data_labels
        )


#
#
# TrainAndTestDatabase classes


class TrainAndTestDatabase(LabeledDatabase[U, V]):
    def __init__(
        self,
        maximum_train_size: Optional[int] = None,
        maximum_test_size: Optional[int] = None,
    ):
        maximum_size: Optional[int] = None
        if maximum_train_size is not None and maximum_test_size is not None:
            maximum_size = maximum_train_size + maximum_test_size
        super().__init__(maximum_size)
        self._maximum_train_size = maximum_train_size
        self._maximum_test_size = maximum_test_size

    @abstractmethod
    def get_original_train_and_test_data(
        self,
    ) -> Tuple[
        Tuple[SimpleSequence[U], SimpleSequence[V]],
        Tuple[SimpleSequence[U], SimpleSequence[V]],
    ]:
        ...

    def get_train_and_test_data(
        self,
    ) -> Tuple[
        Tuple[SimpleSequence[U], SimpleSequence[V]],
        Tuple[SimpleSequence[U], SimpleSequence[V]],
    ]:
        (
            (train_data_set, train_data_labels),
            (test_data_set, test_data_labels),
        ) = self.get_original_train_and_test_data()
        return (
            (
                train_data_set[: self._maximum_train_size],
                train_data_labels[: self._maximum_train_size],
            ),
            (
                test_data_set[: self._maximum_test_size],
                test_data_labels[: self._maximum_test_size],
            ),
        )

    def get_copy_of_train_and_test_data(
        self,
    ) -> Tuple[
        Tuple[SimpleSequence[U], SimpleSequence[V]],
        Tuple[SimpleSequence[U], SimpleSequence[V]],
    ]:
        return self.get_train_and_test_data()


class PreprocessedTrainAndTestDatabase(TrainAndTestDatabase[U, V]):
    def __init__(
        self,
        processed_data_set: SimpleSequence[U],
        processed_data_labels: SimpleSequence[V],
        test_size_percentage: float = 0.25,
        maximum_train_size: Optional[int] = None,
        maximum_test_size: Optional[int] = None,
    ):
        super().__init__(maximum_train_size, maximum_test_size)
        self._processed_data_set = list(processed_data_set)
        self._processed_data_labels = list(processed_data_labels)
        self._test_size_percentage = test_size_percentage

    @staticmethod
    def _split_data_into_train_and_test(
        data: list[V], test_size: float
    ) -> Tuple[list[V], list[V]]:
        random.shuffle(data)
        split_index = round((1 - test_size) * len(data))
        train_data = data[:split_index]
        test_data = data[split_index:]
        return train_data, test_data

    def get_original_train_and_test_data(
        self,
    ) -> Tuple[Tuple[list[U], list[V]], Tuple[list[U], list[V]],]:
        (
            train_data_set,
            test_data_set,
        ) = PreprocessedTrainAndTestDatabase._split_data_into_train_and_test(
            self._processed_data_set, self._test_size_percentage
        )
        (
            train_data_labels,
            test_data_labels,
        ) = PreprocessedTrainAndTestDatabase._split_data_into_train_and_test(
            self._processed_data_labels, self._test_size_percentage
        )
        return (
            (train_data_set, train_data_labels),
            (test_data_set, test_data_labels),
        )

    def get_data(self) -> list[U]:
        return self._processed_data_set

    def get_data_and_labels(
        self,
    ) -> Tuple[list[U], list[V]]:
        return self._processed_data_set, self._processed_data_labels

    def get_copy_of_data(self) -> list[U]:
        return self._processed_data_set.copy()

    def get_copy_of_data_and_labels(
        self,
    ) -> Tuple[list[U], list[V]]:
        return (
            self._processed_data_set.copy(),
            self._processed_data_labels.copy(),
        )


class PreparedTrainAndTestDatabase(TrainAndTestDatabase[U, V]):
    def get_data(self) -> list[U]:
        (
            (train_data_set, _train_data_labels),
            (test_data_set, _test_data_labels),
        ) = self.get_train_and_test_data()

        data_set = list(train_data_set) + list(test_data_set)
        return data_set

    def get_data_and_labels(
        self,
    ) -> Tuple[list[U], list[V]]:
        (
            (train_data_set, train_data_labels),
            (test_data_set, test_data_labels),
        ) = self.get_train_and_test_data()

        data_set = list(train_data_set) + list(test_data_set)
        data_labels = list(train_data_labels) + list(test_data_labels)
        return data_set, data_labels

    def get_copy_of_data(self) -> list[U]:
        return self.get_data().copy()

    def get_copy_of_data_and_labels(
        self,
    ) -> Tuple[list[U], list[V]]:
        data_set, data_labels = self.get_data_and_labels()
        return data_set.copy(), data_labels.copy()


class RawTrainAndTestDatabase(PreprocessedTrainAndTestDatabase[U, V]):
    def __init__(
        self,
        raw_data_set: SimpleSequence,
        raw_data_labels: SimpleSequence,
        feature_extractor: FeatureExtractor,
        label_extractor: LabelExtractor = DEFAULT_LABEL_EXTRACTOR,
        test_size_percentage: float = 0.25,
        maximum_train_size: Optional[int] = None,
        maximum_test_size: Optional[int] = None,
    ):
        self._raw_data_set = as_list(raw_data_set)
        processed_data_set: list[U] = list(
            feature_extractor.extract(self._raw_data_set)
        )
        self._raw_data_labels = as_list(raw_data_labels)
        processed_data_labels: list[V] = list(
            label_extractor.extract(self._raw_data_labels)
        )

        super().__init__(
            processed_data_set,
            processed_data_labels,
            test_size_percentage,
            maximum_train_size,
            maximum_test_size,
        )
        self._feature_extractor = feature_extractor
        self._label_extractor = label_extractor
