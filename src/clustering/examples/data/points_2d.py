from typing import Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sklearn.datasets
from typing_extensions import override

from ...algorithms.util.data_structures.clustering import ArrayBasedClustering
from ..classification.database import PreprocessedLabeledDatabase


class Points2dDatabase(
    PreprocessedLabeledDatabase[
        # npt.NDArray[np.floating],
        npt.NDArray[np.number],
        npt.NDArray[np.integer],
    ]
):
    def __init__(
        self,
        points_data_set: npt.NDArray[np.floating],
        points_data_labels: npt.NDArray[np.integer],
        maximum_size: Optional[int] = None,
    ):
        super().__init__(points_data_set, points_data_labels, maximum_size)

    @override
    def get_data(self) -> npt.NDArray[np.floating]:
        points_data_set = cast(npt.NDArray[np.floating], super().get_data())
        return points_data_set

    @override
    def get_data_and_labels(
        self,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        points_data_set = cast(
            npt.NDArray[np.floating], self._processed_data_set
        )
        points_data_labels = cast(
            npt.NDArray[np.integer], self._processed_data_labels
        )
        return points_data_set, points_data_labels

    def get_copy_of_data(self) -> npt.NDArray[np.floating]:
        return self.get_data().copy()

    def get_copy_of_data_and_labels(
        self,
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.integer]]:
        points_data_set, points_data_labels = self.get_data_and_labels()
        return points_data_set.copy(), points_data_labels.copy()

    def draw(
        self,
        text_above_graph: Optional[str] = None,
        text_below_graph: Optional[str] = None,
    ) -> None:
        points_data_set, points_data_labels = self.get_data_and_labels()

        possible_labels = np.unique(points_data_labels).tolist()
        colors = [
            'tab:blue',
            'tab:orange',
            'tab:green',
            'tab:red',
            'tab:purple',
            'tab:brown',
            'tab:pink',
            'tab:gray',
            'tab:olive',
            'tab:cyan',
        ]
        plt.figure()
        for k, label in enumerate(possible_labels):
            labeled_points = points_data_set[points_data_labels == label, :]
            color = colors[k % len(colors)]
            plt.scatter(
                labeled_points[:, 0],
                labeled_points[:, 1],
                c=color,
                marker='.',
                s=10,
            )
        plt.axis('equal')
        # plt.axis('scaled')
        plt.title(text_above_graph, fontsize='large', fontweight='bold')
        plt.xlabel(text_below_graph, fontsize='large')
        plt.show()


class BlobsDatabase(Points2dDatabase):
    def __init__(self, number_of_blobs: int) -> None:
        (
            points_data_set,
            points_data_labels,
            *_other_values,
        ) = sklearn.datasets.make_blobs(
            n_samples=1000,
            n_features=2,
            centers=number_of_blobs,
            cluster_std=0.6,
            random_state=0,
        )
        super().__init__(points_data_set, points_data_labels)
        self._number_of_blobs = number_of_blobs

    @property
    def number_of_blobs(self) -> int:
        return self._number_of_blobs


class CirclesAndMoonsDatabase(Points2dDatabase):
    def __init__(self) -> None:
        circles_points, circles_labels = sklearn.datasets.make_circles(
            n_samples=1000, factor=0.4, noise=0.05, random_state=0
        )
        moons_points, moons_labels = sklearn.datasets.make_moons(
            n_samples=1000, noise=0.05, random_state=0
        )

        number_of_circles = len(set(circles_labels))
        number_of_moons = len(set(moons_labels))
        circles_points += np.abs(np.min(circles_points, axis=0))
        angle = 1 / 2 * np.pi
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        moons_points = moons_points @ rotation_matrix

        moons_points += np.abs(np.min(moons_points, axis=0)) + np.abs(
            [np.max(circles_points[:, 0]) * 1.05, 0]
        )
        circles_points += [
            0,
            (np.max(moons_points[:, 1]) - np.max(circles_points[:, 1])) / 2,
        ]
        moons_labels += number_of_circles
        points_data_set = np.concatenate((circles_points, moons_points))
        points_data_labels = np.concatenate((circles_labels, moons_labels))

        super().__init__(points_data_set, points_data_labels)
        self._number_of_blobs = number_of_circles + number_of_moons

    @property
    def number_of_blobs(self) -> int:
        return self._number_of_blobs


class BlobAndLineDatabase(Points2dDatabase):
    def __init__(self) -> None:
        random_state = np.random.RandomState(0)

        line_points = random_state.uniform(
            low=[0, -0.01], high=[10, 0.01], size=(1000, 2)
        )
        line_labels = np.full(1000, 0)
        blob_points = random_state.normal(
            # loc=[3, 0.4], scale=[0.3, 0.18], size=(1000, 2)
            # loc=[3, 1], scale=[0.2, 0.18], size=(1000, 2)
            # loc=[3, 0.6], scale=[0.2, 0.18], size=(1000, 2)
            loc=[3, 0.4],
            scale=[0.2, 0.18],
            size=(1000, 2),
        )
        blob_labels = np.full(1000, 1)

        points_data_set = np.concatenate((line_points, blob_points))
        points_data_labels = np.concatenate((line_labels, blob_labels))
        shuffle_random_state = np.random.RandomState(1)
        shuffle_random_state.shuffle(points_data_set)
        shuffle_random_state = np.random.RandomState(1)
        shuffle_random_state.shuffle(points_data_labels)

        super().__init__(points_data_set, points_data_labels)
        self._number_of_blobs = 2

    @property
    def number_of_blobs(self) -> int:
        return self._number_of_blobs


class OverlappedBlobsDatabase(Points2dDatabase):
    def __init__(self) -> None:
        random_state = np.random.RandomState(0)

        blob1_points = random_state.uniform(
            low=[0, 0],
            high=[1, 1],
            size=(300, 2)
            # loc=[0.5, 0.5], scale=[1, 1], size=(300, 2)
        )
        blob1_labels = np.full(300, 0)
        blob2_points = random_state.normal(
            loc=[0.5, 0.5], scale=[0.02, 0.02], size=(500, 2)
        )
        blob2_labels = np.full(500, 1)

        points_data_set = np.concatenate((blob1_points, blob2_points))
        points_data_labels = np.concatenate((blob1_labels, blob2_labels))

        super().__init__(points_data_set, points_data_labels)
        self._number_of_blobs = 2

    @property
    def number_of_blobs(self) -> int:
        return self._number_of_blobs


class EightPointsExample(Points2dDatabase):
    def __init__(self) -> None:
        points_data_set = np.array(
            [
                [1.0, 2.6],
                [1.4, 1.5],
                [1.8, 2.4],
                [2.4, 0.5],
                [3.1, 0.8],
                [3.4, 1.6],
                [4.0, 2.0],
                [4.4, 1.4],
            ]
        )
        points_data_labels = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        # points_data_labels = np.array([0, 0, 0])

        super().__init__(points_data_set, points_data_labels)
        self._number_of_blobs = 3

    @property
    def number_of_blobs(self) -> int:
        return self._number_of_blobs



def draw_points_2d_clustering(
    points_2d_clustering: ArrayBasedClustering[npt.NDArray[np.number]],
    text_above_graph: Optional[str] = None,
    text_below_graph: Optional[str] = None,
) -> None:
    points_2d_clustering_database = Points2dDatabase(
        points_2d_clustering.elements, np.asarray(points_2d_clustering.labels)
    )
    points_2d_clustering_database.draw(text_above_graph, text_below_graph)
