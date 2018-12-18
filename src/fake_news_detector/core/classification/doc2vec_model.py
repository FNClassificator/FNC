from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
import gensim
from sklearn.model_selection import train_test_split

def get_corpus(text2train):
    dictionary = Dictionary(text2train)
    corpus= [dictionary.doc2bow(text) for text in text2train]
    return dictionary, corpus

## TOPIC MODELING

def get_model(var, corpus, n_topics, dictionary):
    switcher = {
        'LSI': LsiModel(corpus=corpus, num_topics=n_topics, id2word=dictionary),
        'HDP': HdpModel(corpus=corpus, id2word=dictionary),
        'LDA':  LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary)
    }
    return switcher.get(var, -1)


class Doc2VecModel():

    def __init__(self, data):
        res = get_corpus(text)
        self.dictionary = res[0]
        self.corpus = res[1]
        self.model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

    def show_topics(self, n):
        self.model.show_topics(num_topics=n)

    

# For evaluate each classificator
# https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/gensim_news_classification.ipynb