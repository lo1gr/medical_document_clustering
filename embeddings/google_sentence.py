# Libraries

import pandas as pd

import tensorflow_hub as hub
import tensorflow.compat.v1 as tf

from embeddings.embedder import Embedder


class GoogleSentence(Embedder):

    def __init__(self):
        super().__init__('google_sentence')

        tf.disable_v2_behavior()

        self.output_format = {
            'n_cols': 512
        }

        self.trained_model = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

    def embed_use_tf(self, module):
        """
        :param module: url of the module
        :return: session
        """
        with tf.Graph().as_default():
            sentences = tf.placeholder(tf.string)
            embed = hub.Module(module)
            embeddings = embed(sentences)
            session = tf.train.MonitoredSession()
        return lambda x: session.run(embeddings, {sentences: x})

    def embed_text(self, abstracts):
        """
        :param abstracts: pandas Series of abstracts
        :return: embedding and associated format
        """

        embed_fn = self.embed_use_tf(self.trained_model)

        embedding = pd.DataFrame(embed_fn(abstracts.tolist()))

        return embedding, self.output_format






