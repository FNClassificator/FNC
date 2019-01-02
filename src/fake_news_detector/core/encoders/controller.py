from src.fake_news_classificator.core.data_process import vectorize as v

""" 
Vectorize Text to BOW
@corpus: Corpus is all the text to encode
@filter_by_freq: If true, dictionary only includes most frequent words
return: all documents encoded by BOW
"""
def encode_by_BOW


def LDA_topic_inspection(dataset, corpus, dictionary, labels):
        model = gensim.models.LdaMulticore(corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
        



