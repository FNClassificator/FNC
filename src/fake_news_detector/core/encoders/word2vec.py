
from gensim.models import Word2Vec

#####################
# CREATE MODEL
#####################


# size: Como menos datos, menor valor
# window: Max distancia entre vecinos
# min_count: Minima frecuencia
# workers: threads

def createWord2Vec(documents, size=150, window=10, min_count=2, workers=10, iterations=6):
    return gensim.models.Word2Vec(documents,
                                size=size,
                                window=window,
                                min_count=min_count,
                                workers=workers,
                                iter=iterations)
#####################
# TRAIN MODEL
#####################


def train(model, length):
    model.train(documents, total_examples=length, epochs=10)

#####################
# EVALUATION MODEL
#####################

def get_top_similars(model, word,  topn=5):
    return model.wv.most_similart(positive=word, topn=topn)

def print_similars(model, word)_
    top_w = get_top_similars(model, word)
    for elem in top_w:
        print('Word:' , elem[0], 'Score:', elem[1])