from clustering.algorithms.partitional.spectral_clustering import (
    ImageDirectedSpectralClustering,
)
from clustering.algorithms.util.kernel import (
    DynamicWidthGaussianKernel,
    LocalDynamicWidthGaussianKernel,
)
from clustering.examples.data.images import CupImage, show_image_segmentation

cup_image = CupImage()

# baseball_image.show()

cup_image_data = cup_image.get_copy_of_data()

directed_spectral_clustering = ImageDirectedSpectralClustering(
    cup_image_data,
    7,
    # intensity_kernel=DynamicWidthGaussianKernel(0.08, 35),
    # spatial_location_kernel=LocalDynamicWidthGaussianKernel(0.03, 35, 0.1),
    # intensity_kernel=DynamicWidthGaussianKernel(0.08, 10),
    # spatial_location_kernel=LocalDynamicWidthGaussianKernel(0.03, 10, 0.1),
    intensity_kernel=DynamicWidthGaussianKernel(0.1, 20),
    spatial_location_kernel=LocalDynamicWidthGaussianKernel(0.08, 20, 0.5),
    # intensity_kernel=DynamicWidthGaussianKernel(0.08 * 10, 10),
    # intensity_kernel=DynamicWidthGaussianKernel(0.8, 10),
    # spatial_location_kernel=LocalDynamicWidthGaussianKernel(0.4, 10, 0.1),
    # intensity_kernel=DynamicWidthGaussianKernel(0.8, 2),
    # spatial_location_kernel=LocalDynamicWidthGaussianKernel(0.8, 20, 0.1),
    # intensity_kernel=DynamicWidthGaussianKernel(0.8, 20),
    # spatial_location_kernel=LocalDynamicWidthGaussianKernel(0.8, 20, 0.2),
    # intensity_kernel=DynamicWidthGaussianKernel(0.4, 20),
    # spatial_location_kernel=LocalDynamicWidthGaussianKernel(0.3, 20, 0.1),
)


directed_spectral_clustering.analyze()

show_image_segmentation(directed_spectral_clustering)

X = 1
