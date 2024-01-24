from clustering.algorithms.partitional.spectral_clustering import (
    UndirectedSpectralClustering,
)
from clustering.examples.data.points_2d import (
    BlobsDatabase,
    draw_points_2d_clustering,
)

blobs_database = BlobsDatabase(4)

# blobs_database.draw()

blobs_data_set, blobs_data_labels = blobs_database.get_data_and_labels()

undirected_spectral_clustering = UndirectedSpectralClustering(
    blobs_data_set, blobs_database.number_of_blobs
)

undirected_spectral_clustering.analyze()

draw_points_2d_clustering(undirected_spectral_clustering)

X = 1
