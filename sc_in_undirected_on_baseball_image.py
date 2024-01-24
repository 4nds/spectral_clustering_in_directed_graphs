from clustering.algorithms.partitional.spectral_clustering import (
    ImageUndirectedSpectralClustering,
)

# pylint: disable=unused-import
from clustering.algorithms.util.kernel import (
    ConstantWidthGaussianKernel,
    ConstantWidthImageGaussianKernel,
    LocalConstantWidthGaussianKernel,
    LocalConstantWidthImageGaussianKernel,
)
from clustering.examples.data.images import (
    BaseballImage,
    show_image_segmentation,
)

# pylint: enable=unused-import

baseball_image = BaseballImage()

# baseball_image.show()

# intensity_kernel_width_percentages = [0.3]
# intensity_kernel_width_percentages = [0.1]
# intensity_kernel_width_percentages = [0.001]


# spatial_kernel_width_percentages = [0.04]
# spatial_kernel_width_percentages = [0.1]
# spatial_kernel_width_percentages = [0.03]

# neighborhood_radius_percentages = [0.5]
# neighborhood_radius_percentages = [0.05]

baseball_image_data = baseball_image.get_copy_of_data()

undirected_spectral_clustering = ImageUndirectedSpectralClustering(
    baseball_image_data,
    7,
    intensity_kernel=ConstantWidthGaussianKernel(0.08),
    spatial_location_kernel=LocalConstantWidthGaussianKernel(0.03, 0.1),
)

# undirected_spectral_clustering = ImageUndirectedSpectralClustering(
#     baseball_image_data,
#     7,
#     intensity_kernel=ConstantWidthImageGaussianKernel(0.01),
#     spatial_location_kernel=LocalConstantWidthImageGaussianKernel(4, 10),
# )

undirected_spectral_clustering.analyze()

show_image_segmentation(undirected_spectral_clustering)

X = 1
