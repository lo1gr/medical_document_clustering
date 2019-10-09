# Libraries

from modeling.clustering_model import ClusteringModel
from sklearn.cluster import AffinityPropagation


class AffinityPropagationModel(ClusteringModel):

    def __init__(self, **params):
        super().__init__('affinity')
        self.model = AffinityPropagation(**params)

