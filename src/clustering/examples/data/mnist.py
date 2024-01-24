from typing import Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf

from ..classification.database import PreparedTrainAndTestDatabase


class MnistDatabase(PreparedTrainAndTestDatabase[npt.NDArray, np.integer]):
    def get_original_train_and_test_data(
        self,
    ) -> Tuple[
        Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]],
        Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]],
    ]:
        train_data_set: npt.NDArray[np.integer]
        train_data_labels: npt.NDArray[np.integer]
        test_data_set: npt.NDArray[np.integer]
        test_data_labels: npt.NDArray[np.integer]
        (
            (train_data_set, train_data_labels),
            (test_data_set, test_data_labels),
        ) = tf.keras.datasets.mnist.load_data()
        return (
            (train_data_set, train_data_labels),
            (test_data_set, test_data_labels),
        )
