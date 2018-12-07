from gensim.models import word2vec

def get_word2vec(sentences):
    model = Word2Vec(sentences, size=200) 