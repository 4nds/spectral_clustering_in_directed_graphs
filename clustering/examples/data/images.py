import math
import os
from typing import Optional, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing_extensions import override

from ...algorithms.util.data_structures.clustering import ImagePixelsClustering
from ..classification.database import PreprocessedDatabase


class ImageDatabase(PreprocessedDatabase[npt.NDArray[np.int_]]):
    def __init__(self, image_data: npt.NDArray[np.int_]) -> None:
        super().__init__(image_data)

    @override
    def get_data(self) -> npt.NDArray[np.int_]:
        image_data = cast(npt.NDArray[np.int_], super().get_data())
        return image_data

    def get_copy_of_data(self) -> npt.NDArray[np.int_]:
        return self.get_data().copy()

    def show(self) -> None:
        plt.imshow(self.get_data())
        plt.show()


class BaseballImage(ImageDatabase):
    def __init__(self) -> None:
        image_filename = 'baseball_100.jpg'
        # image_filename = 'chrome_logo.jpg'
        image_filename = 'baseball_40.jpg'
        bgr_image_data = cv2.imread(
            os.path.join(os.path.dirname(__file__), image_filename)
        )
        image_data = cv2.cvtColor(bgr_image_data, cv2.COLOR_BGR2RGB)
        super().__init__(image_data)


class ZebraImage(ImageDatabase):
    def __init__(self) -> None:
        image_filename = 'zebra_100.jpg'
        bgr_image_data = cv2.imread(
            os.path.join(os.path.dirname(__file__), image_filename)
        )
        image_data = cv2.cvtColor(bgr_image_data, cv2.COLOR_BGR2RGB)
        super().__init__(image_data)


class CupImage(ImageDatabase):
    def __init__(self) -> None:
        image_filename = 'cup_100.jpg'
        bgr_image_data = cv2.imread(
            os.path.join(os.path.dirname(__file__), image_filename)
        )
        image_data = cv2.cvtColor(bgr_image_data, cv2.COLOR_BGR2RGB)
        super().__init__(image_data)


class AppleImage(ImageDatabase):
    def __init__(self) -> None:
        image_filename = 'apple.jpg'
        bgr_image_data = cv2.imread(
            os.path.join(os.path.dirname(__file__), image_filename)
        )
        image_data = cv2.cvtColor(bgr_image_data, cv2.COLOR_BGR2RGB)
        super().__init__(image_data)


def show_image_segmentation(
    image_clustering: ImagePixelsClustering,
    neutral_color: Optional[npt.ArrayLike] = None,
) -> None:
    image_data = image_clustering.elements
    image_intensity = cv2.cvtColor(image_data, cv2.COLOR_RGB2GRAY)
    third_dimension = (image_data.shape + (1,))[2]
    if neutral_color is None:
        if third_dimension == 3:
            # light blue color
            neutral_color = np.array([135, 206, 235])
        else:
            neutral_color = np.full(third_dimension, 255)
    possible_labels = set(image_clustering.labels)
    label_masks = [
        image_clustering.image_shaped_labels == label
        for label in possible_labels
    ]
    image_parts = []
    for label_mask in label_masks:
        color = neutral_color
        if np.all(image_intensity[label_mask] > 20):
            color = np.full(third_dimension, 0)
        if np.all(image_intensity[label_mask] < 230):
            color = np.full(third_dimension, 255)
        image_parts.append(
            np.full(image_data.shape, color, dtype=image_data.dtype)
        )

    for label_mask, image_part in zip(label_masks, image_parts):
        image_part[label_mask] = image_data[label_mask]

    image_parts.insert(0, image_data)

    _figure, axes = plt.subplots(math.ceil(image_clustering.size / 4), 4)

    # pylint: disable-next=invalid-name
    for ax in axes.ravel():
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])

    letters = 'abcdefghijklmnopqrstuvwxyz'
    for i, image_part in enumerate(image_parts):
        column, row = i // 4, i % 4
        axes[column, row].imshow(image_part)
        # axes[column, row].get_xaxis().set_visible(False)
        # axes[column, row].get_yaxis().set_visible(False)
        # axes[column, row].set_axis_off()
        axes[column, row].text(
            0.5,
            -0.3,
            f'{letters[i]})',
            size=12,
            ha='center',
            transform=axes[column, row].transAxes,
        )

    # plt.imshow(image_part)
    _figure.tight_layout(h_pad=0.0, w_pad=2.0)
    # plt.axis('scaled')
    plt.show()
    x = 1


#
