# Libraries

from modeling.clustering_model import ClusteringModel
from sklearn.cluster import MeanShift


class MeanShiftModel(ClusteringModel):

    def __init__(self, **params):
        super().__init__('meanshift')
        self.model = MeanShift(**params)

