import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sklearn.datasets

X: npt.NDArray[np.floating]
y: npt.NDArray[np.integer]

X, y_true, *_other_values = sklearn.datasets.make_blobs(
    n_samples=1000, n_features=2, centers=4, cluster_std=0.6, random_state=0
)

plt.figure(1)
colors = ['#4EACC5', '#FF9C34', '#4E9A06', 'm']

for k, col in enumerate(colors):
    cluster_data = y_true == k
    plt.scatter(
        X[cluster_data, 0], X[cluster_data, 1], c=col, marker='.', s=10
    )

plt.show()

A = 1
