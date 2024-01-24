from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...util.typing_ import SimpleSequence


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, raw_data_set: SimpleSequence) -> SimpleSequence:
        ...


class LabelExtractor(ABC):
    @abstractmethod
    def extract(self, raw_data_labels: SimpleSequence) -> SimpleSequence:
        ...


class DefaultLabelExtractor(LabelExtractor):
    def extract(self, raw_data_labels: SimpleSequence) -> SimpleSequence:
        return raw_data_labels


DEFAULT_LABEL_EXTRACTOR = DefaultLabelExtractor()
