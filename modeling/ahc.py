# Libraries

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from modeling.clustering_model import ClusteringModel

class AHCModel(ClusteringModel):

    def __init__(self, n_clusters=100, affinity="cosine", linkage="ward"):
        super().__init__('ahc')

        self.model = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)

    def perform_clustering(self, features, **params):
        self.model.fit(features, **params)

        return pd.concat(
            [
                features,
                pd.DataFrame(
                    [i for i in self.model.fit_predict(features)],
                    columns=('cluster',))
            ],
            axis=1
        )
