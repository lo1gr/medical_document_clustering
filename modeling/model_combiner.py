# Libraries
import pandas as pd
import numpy as np
import time
from collections import Counter

from modeling.clustering_model import ClusteringModel


class ClusterLabelsCombiner:

    def __init__(self, models_vectors_mapping):
        """
        :param models_vectors_mapping: list, association of ClusteringModel with a specific embedding

        Example: ClusterLabelsCombiner(
            [
                (KMeansModel(n_clusters=15), biowordvec_embedding),
                (DBSCANModel(eps=0.1), bert_embedding)
            ]
        )

        """
        self.models_vectors_mapping = models_vectors_mapping

        self.clusters = self.fit_models()

        self.labels = pd.DataFrame()

    def fit_models(self):
        """
        :return: list with fitted models associated with the used embedding
        """
        clusters = []

        for model, vectors in self.models_vectors_mapping:
            print('Fitting {method}'.format(method=model.model_type))
            start = time.time()
            clusters.append(model.perform_clustering(vectors))
            print('Model {method} fitted in {time} s'.format(method=model.model_type, time=time.time() - start))
        return clusters

    def get_clusters_from_models(self, abstracts):
        """
        :return: list of pandas Series with predicted clusters for each embedding
        """
        return [pd.Series(model.label_clusters(
            clusters=self.clusters[i],
            abstracts=abstracts
        ).labels) for i, (model, vectors) in enumerate(self.models_vectors_mapping)]

    def concat_labels(self, labels):
        """
        Concat a list of multiple labels columns to only one by adding labels in a single group
        :param labels: list of labels columns
        :return: Series, list of a labels column
        """
        result = labels[0]

        for l in range(1, len(labels)):
            result += labels[l]

        return result

    def tfidf(self, labels, number_of_tags_to_keep=5):
        """
        :param labels: Series of labels
        :param number_of_tags_to_keep: Number of tags to keep after the tfidf process
        :return: List of the most relevant labels
        """

        counters = []

        # Term Frequency

        for words in labels.tolist():
            c = Counter(words)
            for k in c.keys():
                c[k] = c[k] / len(words)
            counters.append(c)

        # Inverse Document Frequency

        docs_with_word = Counter()

        for i, l in enumerate(labels.tolist()):
            for word in counters[i].keys():
                docs_with_word[word] += 1

        for k in docs_with_word.keys():
            docs_with_word[k] = np.log(labels.shape[0] / docs_with_word[k])

        # TF-IDF

        for counter in counters:
            for k in counter.keys():
                counter[k] = counter[k] * docs_with_word[k]

        most_relevant_words = []

        for counter in counters:
            most_relevant_words.append([word[0] for word in counter.most_common(number_of_tags_to_keep)])

        return most_relevant_words

    def combine(self, abstracts, number_of_tags_to_keep=5):
        """
        Final function to call
        :return:
        """
        clusters = self.get_clusters_from_models(abstracts=abstracts)

        labels = self.concat_labels(clusters)

        labels = self.tfidf(labels=labels, number_of_tags_to_keep=number_of_tags_to_keep)

        self.labels['labels'] = labels

        return self

    def evaluate(self, embedder, abstracts):
        """
        :param embedder: Embedder used to embed labels
        :param abstracts: original abstracts
        :return: RMSE computed
        """
        labelled_clusters = pd.concat([self.labels, abstracts], axis=1)

        return ClusteringModel.evaluate_clusters(embedder=embedder, labelled_clusters=labelled_clusters)
