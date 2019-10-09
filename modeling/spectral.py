# Libraries

from modeling.clustering_model import ClusteringModel
from sklearn.cluster import SpectralClustering


class SpectralModel(ClusteringModel):

    def __init__(self, n_clusters=10):
        super().__init__('spectral')

        self.model = SpectralClustering(n_clusters=n_clusters, random_state=self.random_state)
