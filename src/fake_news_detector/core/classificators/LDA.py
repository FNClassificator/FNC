
from gensim.models.ldamodel import LdaModel
from sklearn.metrics import classification_report, confusion_matrix  
import pyLDAvis

########### LDA FOR TOPIC 

###############
# CREATE MODEL 
###############

def create_LDA(corpus, dictionary, num_topics=2, passes=15):
    return LdaModel(corpus, 
                    num_topics = num_topics, 
                    id2word=dictionary, 
                    passes=15)

###############
# EVALUATION  
###############

def get_topics(model, num_words=5):
    return model.print_topics(num_words=num_words)

def print_topics(model, num_words):
    topics =  model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)

def get_document_topic(model, document):
    return model.get_document_topic(document)

def print_doc_topics(model, document):
    topics = get_doc_topics(model,document)
    print(topic)

def get_top_words_by_id(lda_model, topic_id, n_top=5):
    id_tuples = lda_model.get_topic_terms(topic_id, topn=n_top)
    word_ids = np.array(id_tuples)[:,0]
    words = map(lambda id_: lda_model.id2word[id_], word_ids)
    return words

###############
# Y EVALUATION  
###############


def get_doc_max_topic(model, document):
    topics = get_document_topic(model, document)
    max_topic = topics[0][0]
    max_value = topics[0][1]

    for topic in topics:
        if max_value < topic[1]:
            max_value = topic[1]
            max_topic = topic[0]

    return max_topic

def get_all_topic_predictions(model, documents):
    results = []
    for document in documents:
        id_topic = get_doc_max_topic(model, document)
        result.append(id_topic)
    return result

def get_evaluation(y_test, y_pred):
    confusion_m = confusion_matrix(y_test,y_pred)
    class_report = classification_report(y_test,y_pred)
    return confusion_m, class_report


def print_evaluation(y_test, y_pred):
    confusion_m, class_report = get_evaluation(y_test, y_pred)
    print('Confusion matrix:')
    print(confusion_m)
    print('REPORT:')
    print(class_report)

###############
# VISUALISE  
###############

def display_modeling(lda,dictionary, coprus):
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)