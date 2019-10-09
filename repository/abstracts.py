import pandas as pd
import numpy as np
import gensim
import nltk
import warnings
import re

from Bio import Entrez
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from gensim.utils import simple_preprocess
from gensim import models
from gensim.parsing.preprocessing import STOPWORDS

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

warnings.filterwarnings("ignore")


def search(query):
    """ search articles with the key word 'query'.
    The database is specified in the parameter 'db'.
    The number of retrived articles is specified in the parameter 'retmax'.
    The reason for declaring YOUR_EMAIL address is to allow the NCBI to
    contact you before blocking your IP, in case you’re violating the guidelines.
    """
    Entrez.email = 'YOUR_EMAIL'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='1000',
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results


def fetch_details(id_list):
    """ Fetch details of a list of articles IDs.
    The reason for declaring YOUR_EMAIL address is to allow the NCBI to
    contact you before blocking your IP, in case you’re violating the guidelines.
    """
    ids = ','.join(id_list)
    Entrez.email = 'YOUR_EMAIL'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results


def collect_data(categories):
    """ Get abstracts for each category in 'categories'.

    return: abstracts
    ----------
    article_ID : ID of the article
    text : abstract of the article if it exists
    category : category of the article
    structured : indicates whether the article is structured
    Keywords : keywords of the article if it exists
    Title : title of the article
    """
    abstracts = pd.DataFrame(columns=['article_ID', 'Title',
                                      'Keywords', 'text', 'category',
                                      'structured'])
    for cat in categories:
        results = search(cat)  # get the articles for the category 'cat'
        id_list = results['IdList']  # select the IDs
        if (len(id_list) > 0):
            papers = fetch_details(id_list)  # get details of articles
            pubmed_articles = papers['PubmedArticle']
            for pubmed_article in pubmed_articles:
                s = 1  # structured article
                MedlineCitation = pubmed_article['MedlineCitation']
                pmid = int(str(MedlineCitation['PMID']))
                article = MedlineCitation['Article']
                keywords = MedlineCitation["KeywordList"]
                title = MedlineCitation['Article']['ArticleTitle']
                if (len(keywords) > 0):
                    keywords = list(keywords[0])
                if ('Abstract' in article):
                    abstract = article['Abstract']['AbstractText']
                    if (len(abstract) == 1):
                        abstract = abstract[0]
                        s = 0
                else:
                    abstract = ''
                abstracts = abstracts.append({'article_ID': pmid, 'text': abstract,
                                              'category': cat, 'structured': s,
                                              'Keywords': keywords, 'Title': title},
                                             ignore_index=True)  # store the abstract
    return abstracts


# here are defined categories for which we want articles
categories = ['cancérologie', 'cardiologie', 'gastro',
              'diabétologie', 'nutrition', 'infectiologie',
              'gyneco-repro-urologie', 'pneumologie', 'dermatologie',
              'industrie de santé', 'ophtalmologie']

# call the function collect_data to get the abstracts
abstracts = collect_data(categories)

abstracts.to_csv('data/CS2_Article_Clustering.xlsx', index=None)
