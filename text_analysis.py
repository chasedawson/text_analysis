import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import collections
import itertools

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# text_col
def get_words(text_col):
    return [text.strip().split(' ') for text in text_col]

def remove_stop_words(text_words):
    return [list(filter(lambda x: x not in stop_words, words)) for words in text_words]

# look into part of speech tagging with nltk,
# would make lemmatization more powerful
def lemmatize(text_words):
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(word) for word in words] for words in text_words]

def stem(text_words):
    stemmer = PorterStemmer()
    return [[stemmer.stem(word) for word in words] for words in text_words]

# return Pandas DataFrame containing bigrams
def get_bigrams(text_col, top_n = 500, clean = "stem"):
    # extract words from text_col, remove stop words, and lemmatize
    text_words = remove_stop_words(get_words(text_col))

    if clean == "lemmatize":
        text_words = lemmatize(text_words)
    else:
        text_words = stem(text_words)

    terms_bigram = [list(bigrams(words)) for words in text_words]
    bigrams = list(itertools.chain(*terms_bigram))
    bigram_counts = collections.Counter(bigrams)
    bigram_df = pd.DataFrame(bigram_counts.most_common(top_n), columns = ['bigram', 'count'])
    bigram_df['item1'] = bigram_df.bigram.apply(lambda x: x[0])
    bigram_df['item2'] = bigram_df.bigram.apply(lambda x: x[1])
    bigram_df = bigram_df.drop(columns=['bigram'])
    bigram_df = bigram_df[['item1', 'item2', 'count']]
    return bigram_df

# get co-occurrences
def get_cooc(text_col, top_n = 500, clean = "stem"):
    text_words = remove_stop_words(get_words(text_col))

    if clean == "lemmatize":
        text_words = lemmatize(text_words)
    else:
        text_words = stem(text_words)
        
    terms_cooc = [list(itertools.permutations(words, 2)) for words in text_words]
    cooc = list(itertools.chain(*terms_cooc))
    cooc_counts = collections.Counter(cooc)
    cooc_df = pd.DataFrame(cooc_counts.most_common(top_n), columns = ['cooc', 'count'])
    cooc_df['item1'] = cooc_df.cooc.apply(lambda x: x[0])
    cooc_df['item2'] = cooc_df.cooc.apply(lambda x: x[1])
    cooc_df = cooc_df.drop(columns=['cooc'])
    cooc_df = cooc_df[['item1', 'item2', 'count']]
    return cooc_df

