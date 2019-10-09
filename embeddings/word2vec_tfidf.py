# Libraries
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from embeddings.embedder import Embedder

# to download the self trained model: https://stackoverflow.com/questions/46433778/import-googlenews-vectors-negative300-bin
# https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz


class Word2VecTFIDF(Embedder):

    def __init__(self):
        super().__init__('google_sentence')

        self.output_format = {
            'n_cols': 300
        }

        self.trained_model = 'bin/GoogleNews-vectors-negative300.bin'

    # Core functions

    def get_vect(self, word, model):
        """
        :param word: word to be embedded
        :param model: model used to embed a word
        :return: word vector
        """
        try:
            return model.get_vector(word)
        except KeyError:
            return np.zeros((model.vector_size,))

    def sum_vectors(self, sentence, model):
        """
        Extracts every word from the sentence and gets the corresponding vector
        :param sentence: sentence to be embedded
        :param model: model used to embed a word
        :return: vector
        """


        return (self.get_vect(w, model) for w in sentence)

    def word2vec_features(self, sentences, model):
        """
        :param sentences: sentences to be embedded
        :param model: model used to embed a word
        :return: numpy list of list
        """

        for p in sentences:
            for w in p:
                self.sum_vectors(p, model)
        return feats


    def embed_text(self, abstracts, output_format=None):
        """
        :param abstracts: pandas Series of abstracts
        :param output_format: dict specifying output format of the embedding method
        :return: embedding and associated format
        """

        word_vectors = pd.DataFrame(KeyedVectors.load_word2vec_format(self.trained_model, binary=True))



        embedding = pd.DataFrame(self.word2vec_features(abstracts, word_vectors))

        # embedding = pd.DataFrame(self.get_vect(w, word_vectors) for w in abstracts)


        return embedding, output_format


word_vectors = KeyedVectors.load_word2vec_format('bin/GoogleNews-vectors-negative300.bin', binary=True)
# check if veco col word is in document, if it is then

# embed_text(abstracts.nouns_lemmatized_text)


test1 = abstracts.nouns_lemmatized_text[0].replace('\'','').replace('[','').replace(']','').split(',')
test1
test2 = []
for w in test1:
    test2.append(w.strip())


def get_vect(word, model):
    """
    :param word: word to be embedded
    :param model: model used to embed a word
    :return: word vector
    """
    try:
        return model.get_vector(word)
    except KeyError:
        return np.zeros((model.vector_size,))

d = []
for p in test2:
    d.append({'word': p, 'vector': get_vect(p,word_vectors)})

d = pd.DataFrame(d)

for index, i in d.iterrows():
    print(index, i.word)

big_dic = np.array(veco.columns)

# now let's populate the veco:
for index, i in d.iterrows():
    if i.word in big_dic:
        veco.loc[0,i.word] = d.loc[index, i.word]
    else:
        print('no')
        veco[0, i.word] = np.zeros(300)

veco[0, 'proportion']
veco[:,'proportion']


veco.loc[0,'proportion'].to_numpy()
veco.loc[0,'proportion'].to_numpy() = np.zeros(300)
type(veco.loc[0,'proportion'])


type(d.loc[0,'vector'])