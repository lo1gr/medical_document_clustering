# Libraries

import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from embeddings.embedder import Embedder


class BioWordVec(Embedder):

    def __init__(self):
        super().__init__('biowordvec')

        self.output_format = {
            'n_cols': 200
        }

        self.trained_model = 'bin/bio_embedding_extrinsic'

        self.model = KeyedVectors.load_word2vec_format(self.trained_model, binary=True)

    # Core functions

    def get_vect(self, word):
        """
        :param word: word to be embedded
        :param model: model used to embed a word
        :return: word vector
        """
        try:
            return self.model.get_vector(word)
        except KeyError:
            return np.zeros((self.model.vector_size,))

    def sum_vectors(self, sentence):
        """
        :param sentence: sentence to be embedded
        :param model: model used to embed a word
        :return: vector
        """
        return sum(self.get_vect(w) for w in sentence)

    def word2vec_features(self, sentences):
        """
        :param sentences: sentences to be embedded
        :param model: model used to embed a word
        :return: numpy list of list
        """
        feats = np.vstack([self.sum_vectors(p) for p in sentences])
        return feats

    def embed_text(self, abstracts):
        """
        :param abstracts: pandas Series of abstracts
        :param output_format: dict specifying output format of the embedding method
        :return: embedding and associated format
        """

        embedding = pd.DataFrame(self.word2vec_features(abstracts))

        return embedding, self.output_format
