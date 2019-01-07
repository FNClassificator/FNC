from gensim.models.ldamodel import LdaModel
from gensim.models import LsiModel
from sklearn.metrics import classification_report, confusion_matrix  
import pyLDAvis
import pyLDAvis.gensim
import numpy as np

###############
# CREATE MODEL 
###############

def create_LDA(corpus, dictionary, num_topics=2, passes=15):
    return LdaModel(corpus, 
                    num_topics = num_topics, 
                    id2word=dictionary, 
                    passes=15)

def create_LSI(corpus, dictionary, num_topics):
    return LsiModel(corpus, 
                num_topics = num_topics, 
                id2word=dictionary)


###################
# MODEL EVALUATION
###################

def model_evaluation(model, test_docs):
    print(intra_inter(lda_model, test_docs))

###################
# TOPIC EVALUATION  
###################

# Top words by of all topic models
def print_top_words(model):
    for idx, topic in model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

# Top words of a topic
def print_top_words_of_topic(lda_model, topic_id):
    id_tuples = lda_model.get_topic_terms(topic_id, topn=10)
    word_ids = np.array(id_tuples)[:,0]
    print(word_ids)
    words = []
    for tid in word_ids:
        word = lda_model.id2word(tid)
        print(word)
        words.append()
    print('Topic: {} Words: {}'.format(topic_id, words))

# Return topic distribution by document
def get_topic_distribution_of_doc(model, dictionary, document):
    bow_doc = dictionary.doc2bow(document) # Encode 2 bow
    return model.get_document_topics(bow_doc) # Get topics

def get_topics_distribution_by_doc(model, dictionary, corpus):
    result = []
    for doc in corpus:
        topic_dist = get_topic_distribution_of_doc(model, dictionary, doc)
        result.append(topic_dist)
    return result


###############
# VISUALISE  
###############

def display_modeling(lda,dictionary, corpus):
    return pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    