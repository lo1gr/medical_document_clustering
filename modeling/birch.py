# Libraries

from modeling.clustering_model import ClusteringModel
from sklearn.cluster import Birch


class BirchModel(ClusteringModel):

    def __init__(self, n_clusters=10):
        super().__init__('birch')

        self.model = Birch(n_clusters=n_clusters)
