# Libraries

import pandas as pd
from bert_embedding import BertEmbedding

from embeddings.embedder import Embedder


class Bert(Embedder):

    def __init__(self):
        super().__init__('bert')

        self.output_format = {
            'n_cols': 768
        }

        self.model = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased')

    # Core functions

    def embed_text(self, abstracts):
        """
        :param abstracts: pandas Series of abstracts
        :param output_format: dict specifying output format of the embedding method
        :return: embedding and associated format
        """
        bert_embedding = self.model.bert(abstracts.tolist(), oov_way='sum')

        embedding = []

        for _, vectors in bert_embedding:
            embedding.append(sum(vectors))

        embedding = pd.DataFrame(embedding)

        return embedding, self.output_format
