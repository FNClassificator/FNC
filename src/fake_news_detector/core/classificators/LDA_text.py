
from gensim.models.ldamulticore import LdaMulticore


# LDA FUNCTIONS

def create_LDA(corpus, dictionary, num_topics):
    return LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=2)

# GET INFO ABOUT TOPICS
def print_top_words(model):
    for idx, topic in model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))