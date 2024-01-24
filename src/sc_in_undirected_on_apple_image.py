from clustering.algorithms.partitional.spectral_clustering import (
    ImageUndirectedSpectralClustering,
)
from clustering.algorithms.util.kernel import ConstantWidthGaussianKernel
from clustering.examples.data.images import AppleImage, show_image_segmentation

baseball_image = AppleImage()

baseball_image.show()

# intensity_kernel_width_percentages = [0.3]

# baseball_image_data = baseball_image.get_copy_of_data()

# undirected_spectral_clustering = ImageUndirectedSpectralClustering(
#     baseball_image_data,
#     7,
#     intensity_kernel=ConstantWidthGaussianKernel(
#         intensity_kernel_width_percentages[0]
#     ),
# )

# undirected_spectral_clustering.analyze()

# show_image_segmentation(undirected_spectral_clustering)

X = 1
