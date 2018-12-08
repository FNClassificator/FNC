import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from src.fake_news_detector.helpers.nlp import clean_text as ct


# With JACCARD
def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_jaccard_similarity(token_one, token_two):
    words1 = set(token_one)
    words2 = set(token_two)
    return jaccard(words1, words2)


# With COSINE similarity
def get_cosine_similarity(text_one, text_two):
    documents = [text_one, text_two]
    # Translate to vec
    LemVectorizer = CountVectorizer(analyzer=ct.clean_text_words)
    LemVectorizer.fit_transform(documents)
    tf_matrix = LemVectorizer.transform(documents).toarray()
    print(tf_matrix.shape)
    
    # Feature extraction
    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    return tfidfTran.fit(tf_matrix)
