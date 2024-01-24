from clustering.algorithms.partitional.k_means_clustering import (
    get_k_means_clustering,
)
from clustering.examples.data.points_2d import (
    BlobsDatabase,
    draw_points_2d_clustering,
)

blobs_database = BlobsDatabase(4)

# blobs_database.draw()

blobs_data_set, blobs_data_labels = blobs_database.get_data_and_labels()

k_means_clustering = get_k_means_clustering(
    blobs_data_set, blobs_database.number_of_blobs
)

draw_points_2d_clustering(k_means_clustering)
