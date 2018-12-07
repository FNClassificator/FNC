import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def get_cosine_similarity(text_one, text_two):
    documents = [text_one, text_two]
    # Translate to vec
    LemVectorizer = CountVectorizer(stop_words='english')
    LemVectorizer.fit_transform(documents)
    tf_matrix = LemVectorizer.transform(documents).toarray()
    print(tf_matrix.shape)
    
    # Feature extraction
    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    return sim