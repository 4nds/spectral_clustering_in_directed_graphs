from typing import cast

from clustering.algorithms.util.metric import (
    TensorMetric,
    get_tensor_quasi_metric,
)
from clustering.examples.data.mnist import MnistDatabase

mnist_database = MnistDatabase(maximum_train_size=7, maximum_test_size=7)

euclidean_metric = cast(TensorMetric, get_tensor_quasi_metric('euclidean'))


(
    (train_data_set, train_data_labels),
    (test_data_set, test_data_labels),
) = mnist_database.get_train_and_test_data()

element1 = train_data_set[0]
element2 = train_data_set[1]

# element1 = train_data_set[0][15:18, 15:18]
# element2 = train_data_set[1][15:18, 15:18]

# element1 = np.array([[186, 253, 253], [16, 93, 252], [0, 0, 249]])
# element2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 7]])

# element1 = np.array([[186, 253, 253], [16, 93, 252]])
# element2 = np.array([[1j, 1j, 1j], [1j, 1j, 1j]])

# print(euclidean_metric.distance(element1, element2))
# print(np.linalg.norm(element1 - element2))

# print(euclidean_metric.distance(element1, element2, _pairwise=True))
# print(np.linalg.norm(element1 - element2, axis=1))


a, b, c = train_data_set[0:3]
x, y = train_data_set[3:5]

print(euclidean_metric.pairwise_distances_no_memory_limit([a, b, c], [x, y]))
print()
print(euclidean_metric.pairwise_distances([a, b, c], [x, y]))
print()
print(euclidean_metric.distance(a, x))
print(euclidean_metric.distance(c, y))


X = 1
