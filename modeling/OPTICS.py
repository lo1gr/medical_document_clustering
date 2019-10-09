# Libraries

import pandas as pd
from sklearn.cluster import OPTICS
from modeling.clustering_model import ClusteringModel


class OPTICSModel(ClusteringModel):

    def __init__(self, **params):
        super().__init__('optics')

        self.model = OPTICS(**params)

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


