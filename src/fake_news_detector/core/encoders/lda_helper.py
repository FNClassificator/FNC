from gensim import models
from gensim.models import LdaMulticore
import numpy as np

##############################
## TOPIC MODELING WITH LDA ###
##############################

def print_topics(lda_model):
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))

class LDA_model():

    def __init__(self, dictionary, num_topics):
        self.dictionary = dictionary
        self.num_topics = num_topics
        self.lda_model = None

    # LDA Model
    def create_model(self, data):
        return LdaMulticore(data, num_topics=self.num_topics, id2word=self.dictionary, passes=2, workers=2)

    # Transformations
    def get_document_distribution(self,document, min_prob=0):
        topic_importances = self.lda_model.get_document_topics(document, minimum_probability=min_prob)
        topic_importances = np.array(topic_importances)
        return topic_importances[:,1]


    # Getters
    def get_model(self):
        return self.lda_model
    
    def get_dictionary(self):
        return self.dictionary


    # USAGE
    # Create LDA Model
    # Tranform data to LDA
    def run_modeling(self, train_data, test_data):

        # Create LDA model & fit
        self.lda_model = self.create_model(train_data)

        # Get train document distributions
        train_lda_data = list(map(lambda doc:
                                        self.get_document_distribution(doc),
                                        train_data))

        # Get test document distributions
        test_lda_data = list(map(lambda doc:
                                        self.get_document_distribution(doc),
                                        test_data))

        return train_lda_data, test_lda_data

    # MODEL INFO
