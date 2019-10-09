# Libraries

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd
import os
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag


# Mandatory downloads


def nltk_package_downloads():
    """
    function that calls necessary downloads
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


def stopword_list(prep_type):
    """
    :param prep_type: a string containing the type of preproceesing, i.e. "title" or "text"
    :return: a list with stopwords
    """
    # stopwords + lowercase
    normal_stopwords = stopwords.words('english')

    #     import more extensive stopwords + convert to list the first column
    comprehensive_stopwords = pd.read_csv('stopwords/stopwords-en.txt', header=None)[0].tolist()

    #     potential further stopwords (will have to test that)
    special_medical = ["complex", "patients", "treatment", "months",
                       "rate", "prevalence", "case", "early", "management", "data", "factors",
                       "reported", "information", "baseline", "study", "questionnaire", "results", "month", "months",
                       "years", "year", "status", "type", "cells", "cell", "nan",
                        "among", "clinical", "associated", "hospital", "care", "age", "iii", " ", "  ",
                       "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                       "november", "december"]

    if prep_type == "text":
        stopW = normal_stopwords + comprehensive_stopwords + special_medical
        # for the title, we also want to remove individual special words
    else:
        stopW = normal_stopwords + special_medical + comprehensive_stopwords

    return stopW


# word preprocessing
def preprocessing_sentence(column):
    """
    :param column: column to preprocess
    :return: a pre-processed column
    """

    # Tokenization
    tokens = sent_tokenize(str(column))

    tokens = [token.lower().strip() for token in tokens]

    # Deleting specific characters
    special_characters = ["@", "/", "#", ".", ",", "!", "?", "(", ")",
                          "-", "_", "’", "'", "\"", ":", "=", "+", "&",
                          "`", "*", "0", "1", "2", "3", "4", "5",
                          "6", "7", "8", "9", "'", '.', '‘', ';', "%"]
    transformation_sc_dict = {initial: "" for initial in special_characters}
    tokens = [token.translate(str.maketrans(transformation_sc_dict)) for token in tokens]
    return ".".join(tokens)


def preprocessing_words(column, prep_type):
    """
    :param column: column to preprocess
    :param prep_type: a string containing the type of preprocessing, i.e. "title" or "text"
    :return: a pre-processed column
    """

    # Tokenization
    tokens = word_tokenize(str(column))

    # Deleting words with  only one or 2 caracter
    tokens = [token.strip() for token in tokens if len(token.strip()) > 3 and token.find(' ') == -1 and token.find('  ') == -1]


    # Deleting specific characters
    special_characters = ["@", "/", "#", ".", ",", "!", "?", "(", ")",
                          "-", "_", "’", "'", "\"", ":", "=", "+", "&",
                          "`", "*", "0", "1", "2", "3", "4", "5",
                          "6", "7", "8", "9", "'", '.', '‘', ';', '\t']
    transformation_sc_dict = {initial: " " for initial in special_characters}
    tokens = [token.translate(str.maketrans(transformation_sc_dict)) for token in tokens]

    # deleting stopwords
    stopW = stopword_list(prep_type)

    tokens = [token.lower() for token in tokens if token.lower() not in stopW]
    return tokens


def lemmatization(tokenized_column):
    """
    :param tokenized_column: a column of a dataframe that is tokenized
    :return: lemmatized list
    """
    # Lemmatization:
    wnl = WordNetLemmatizer()

    tags = [pos_tag(token) for token in tokenized_column]
    # initialized lemmatizred list:
    lemmatized = []
    nouns_list = []

    for list in tags:
        # initialize empty intermediary liust
        result = []
        nouns = []
        for word, tag in list:
            if word != '    ' and word != '     ' and word != ' ' and word != '  ' and\
                    word != 'p       ' and word!= 'p     ' and word != 'p    ' and word != 'p   ':
                if tag.startswith("NN"):
                    result.append(wnl.lemmatize(word, pos='n'))
                    nouns.append(wnl.lemmatize(word, pos='n'))
                elif tag.startswith("VB"):
                    result.append(wnl.lemmatize(word, pos='v'))
                elif tag.startswith("JJ"):
                    result.append(wnl.lemmatize(word, pos='a'))
                    nouns.append(wnl.lemmatize(word, pos='a'))
                elif tag.startswith("R"):
                    result.append(wnl.lemmatize(word, pos='r'))
                else:
                    result.append(word)
        lemmatized.append(result)
        nouns_list.append(nouns)

    return lemmatized, nouns_list


def detokenize(df_tokens, name_token_column):
    """
    :param df_tokens: dataframe with a tokenized column that will be detokenized
    :param name_token_column: string with the name of the column holding the tokenized data
    :return: the same dataframe with an additional detokenized column called "detokenized"
    the detokenization turns [early, phase, trials, institut, oncology] into early phase therapeutic trials oncology
    """
    df_tokens["detokenized"] = df_tokens[name_token_column].apply(TreebankWordDetokenizer().detokenize)
    return df_tokens


def launch_preprocessing(df):
    """
    :return: a dataframe containing a title and a text column with strings that will be preprocessed (standard NLP preprocessing)
    :return: a preproceessed data frame
    """
    df = df.rename(columns={"Tiltle": "title"})
    df = df.dropna(subset=['text', 'title'])

    df["word_tokens"] = df["text"].apply(preprocessing_words, args=["text"])
    df["sentence_tokens"] = df["text"].apply(preprocessing_sentence)
    df["title_clean"] = df["title"].apply(preprocessing_words, args=["title"])

    #lemmatize:
    df["word_tokens_lemmatized"], df["nouns_lemmatized_text"] = lemmatization(df["word_tokens"])
    df["title_clean_lemmatized"], df["nouns_lemmatized_title"] = lemmatization(df["title_clean"])

    final = detokenize(df, "word_tokens")

    return final


# comprehensive_stopwords = pd.read_csv('stopwords/stopwords-en.txt', header=None)[0].tolist()

# comprehensive_stopwords


def load_data(abstracts_path, with_preprocess=True):
    """
    :param abstracts_path: Path to preprocessed or not abstracts dataset
    :param with_preprocess: Boolean to specify if we should apply preprocess pipeline
    :return: abstracts dataset preprocessed
    """
    abstracts = pd.read_excel(abstracts_path)

    if with_preprocess:

        abstracts = launch_preprocessing(abstracts)

    return abstracts



#abstracts = pd.read_excel('data/CS2_Article_Clustering.xlsx', index=False)
#abstracts = launch_preprocessing(abstracts)

# abstracts.to_csv('data/abstracts_preproc.csv', index=False)
