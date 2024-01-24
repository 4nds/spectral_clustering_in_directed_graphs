import numpy as np

from clustering.algorithms.partitional.spectral_clustering import (
    DirectedSpectralClustering,
)

# pylint: disable=unused-import
from clustering.algorithms.util.kernel import (
    DynamicWidthGaussianKernel,
    LocalDynamicWidthGaussianKernel,
)
from clustering.examples.data.points_2d import (
    EightPointsExample,
    draw_points_2d_clustering,
)

# pylint: enable=unused-import


eight_points_example = EightPointsExample()

eight_points_example_data = eight_points_example.get_copy_of_data()

# eight_points_example.draw()

undirected_spectral_clustering = DirectedSpectralClustering(
    eight_points_example_data,
    3,
    # kernel=DynamicWidthGaussianKernel(0.1, 5),
    kernel=LocalDynamicWidthGaussianKernel(0.2, 1.5, 0.4),
)

weight_matrix = undirected_spectral_clustering.get_weight_matrix()

undirected_spectral_clustering.analyze()

parameters_text = f'f={1.5} σ={0.2}  r={0.4}'

print(np.around(weight_matrix, 2))

draw_points_2d_clustering(
    undirected_spectral_clustering, text_below_graph=parameters_text
)

X = 1
