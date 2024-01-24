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
from clustering.examples.data.images import ZebraImage, show_image_segmentation

# pylint: enable=unused-import


zebra_image = ZebraImage()

# baseball_image.show()


zebra_image_data = zebra_image.get_copy_of_data()

undirected_spectral_clustering = ImageUndirectedSpectralClustering(
    zebra_image_data,
    7,
    intensity_kernel=ConstantWidthGaussianKernel(0.08),
    spatial_location_kernel=LocalConstantWidthGaussianKernel(0.03, 0.1),
)

# undirected_spectral_clustering = ImageUndirectedSpectralClustering(
#     zebra_image_data,
#     7,
#     intensity_kernel=ConstantWidthImageGaussianKernel(0.01),
#     spatial_location_kernel=LocalConstantWidthImageGaussianKernel(4, 10),
# )

undirected_spectral_clustering.analyze()

show_image_segmentation(undirected_spectral_clustering)

X = 1
