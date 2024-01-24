import matplotlib.pyplot as plt
import numpy as np

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

plt.figure()
plt.scatter(
    points_data_set[:, 0],
    points_data_set[:, 1],
    facecolors='none',
    edgecolors='black',
    marker='o',
    s=200,
)
for i, point in enumerate(points_data_set):
    plt.text(
        point[0],
        point[1],
        i + 1,
        horizontalalignment='center',
        verticalalignment='center',
    )

# plt.axis('equal')
plt.xlim(
    np.min(points_data_set[:, 0]) - 0.5, np.max(points_data_set[:, 0]) + 0.5
)
plt.ylim(
    np.min(points_data_set[:, 1]) - 0.5, np.max(points_data_set[:, 1]) + 0.5
)
# pylint: disable-next=invalid-name
ax = plt.gca()
ax.xaxis.set_tick_params(labelbottom=False)
ax.yaxis.set_tick_params(labelleft=False)
ax.set_xticks([])
ax.set_yticks([])
plt.show()
