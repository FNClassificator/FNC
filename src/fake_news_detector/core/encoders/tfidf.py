from gensim import corpora, models


##############################
######## TF  ENCODING ########
##############################


# MAIN FUNCTION ......................................................
def tfidf_encoding(bow_corpus):
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return corpus_tfidf
# ...................................................................


    