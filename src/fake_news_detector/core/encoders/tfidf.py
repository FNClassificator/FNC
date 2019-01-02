from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_encoding():
    tfidf = TfidfVectorizer(sublinear_tf=True, 
                            min_df=5, 
                            norm='l2', 
                            encoding='utf-8', 
                            ngram_range=(1, 2))
                            topic modeling lda