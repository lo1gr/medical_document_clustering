# Libraries
import itertools
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine


class ClusteringModel:

    def __init__(self, model_type):
        self.model_type = model_type
        self.model = KMeans(n_clusters=15)
        self.random_state = 42

    def set_model_parameters(self, **params):
        self.model.set_params(**params)
        return self

    def perform_clustering(self, features, **params):
        self.model.fit(features, **params)

        return pd.concat(
            [
                features,
                pd.DataFrame(
                    [i for i in self.model.predict(features)],
                    columns=('cluster',))
            ],
            axis=1
        )

    def label_clusters(self, clusters, abstracts):
        """
        :param clusters: df with clusters
        :param abstracts: df with all the information
        :return labelled clusters
        """

        clusters["nouns_lemmatized_title"] = abstracts.nouns_lemmatized_title.values
        clusters["nouns_lemmatized_text"] = abstracts.nouns_lemmatized_text.values
        clusters["category"] = abstracts.category

        for k in range(len(pd.unique(clusters.cluster))):
            cluster = clusters.loc[clusters.cluster == k, :]
            words = [y for x in itertools.chain(cluster.nouns_lemmatized_title) for y in x]
            most_common_words = Counter(words).most_common(5)
            print(k)
            print(most_common_words)
            most_common_words = [word[0] for word in most_common_words]
            clusters.loc[clusters.cluster == k, 'labels'] = pd.Series([most_common_words] * len(cluster)).values

        return clusters

    @staticmethod
    def evaluate_clusters(embedder, labelled_clusters):

        embedded_category = np.array(embedder.embed_text(labelled_clusters.nouns_lemmatized_text)[0])
        embedded_labels = np.array(embedder.embed_text(labelled_clusters.labels)[0])

        similarity_vector = []

        for i in range(len(embedded_labels)):
            similarity_vector.append(cosine(embedded_category[i], embedded_labels[i]))

        return np.sqrt(sum([a ** 2 for a in similarity_vector]) / len(similarity_vector))

    def plot_elbow(self, features, range):
        inertia = []

        for r in range:
            self.model.set_params(n_clusters=r)
            self.model.fit(features)
            inertia.append(self.model.inertia_)

        plt.plot(range, inertia)
        plt.show()

    def plot_from_pca(self, clusters):
        """
        :param clusters: df with embedded text and associated cluster
        :return: clusters plot on two first PCA dimensions
        """

        vectors = clusters.drop(["cluster"], axis=1)

        pca = pd.DataFrame(PCA(n_components=2, random_state=self.random_state).fit_transform(
            StandardScaler().fit_transform(
                vectors
            )
        ), columns=('dim_1', 'dim_2'))

        clusters = pd.concat(
            [pca, clusters], axis=1
        )

        plt.scatter(clusters.dim_1, clusters.dim_2, c=clusters.cluster, alpha=0.8)
        plt.show()

    def nb_categories_in_clusters(self, labelled_clusters):
        return len(labelled_clusters.groupby(["cluster", "category"]).count()) / len(pd.unique(labelled_clusters.cluster))

    def concat_clusters_with_abstracts_information(self, clusters, abstracts, columns):
        return pd.concat([clusters, abstracts[columns]], axis=1)
