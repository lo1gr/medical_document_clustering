# Libraries

import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf

from embeddings.embedder import Embedder

# Tutorial:

# https://medium.com/saarthi-ai/elmo-for-contextual-word-embedding-for-text-classification-24c9693b0045


class ELMo(Embedder):

    def __init__(self):
        super().__init__('elmo')

        self.output_format = {
            'n_cols': 1024
        }

        self.trained_model = "https://tfhub.dev/google/elmo/2"

        self.model = hub.Module(self.trained_model, trainable=True)

    def embed_text(self, abstracts):
        """
        :param abstracts: pandas Series of abstracts
        :param output_format: dict specifying output format of the embedding method
        :return: embedding and associated format
        """
        embeddings = self.model(abstracts.tolist(), signature="default", as_dict=True)["elmo"]

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())
            # return average of ELMo features
            return sess.run(tf.reduce_mean(embeddings, 1)), self.output_format
