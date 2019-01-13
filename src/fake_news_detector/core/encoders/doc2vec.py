from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

##############################
#### DOC2VEC Distribution ####
##############################


class Doc2Vec_model():

    def __init__(self, max_epochs, alpha, dm, vector_size, min_count, negative,hs, sample):
        self.max_epochs = max_epochs
        self.alpha = alpha

        self.dm = dm
        self.vector_size = vector_size
        self.min_count = min_count
        self.num_workers = 4
        self.negative = negative
        self.hs = hs
        self.sample = sample
        self.model = None

    # LDA Model
    def create_model(self):
        return  Doc2Vec(dm=self.dm,
                        size = self.vector_size,
                        alpha  = self.alpha,
                        min_alpha = self.min_alpha,
                        min_count = self.min_count,
                        negative = self.negative,
                        hs = self.hs,
                        sample = self.sample,
                        workers = self.num_workers)

    # Transformations


    # Getters
    def get_model(self):
        return self.lda_model
    
    def get_dictionary(self):
        return self.dictionary


    # USAGE
    # Create LDA Model
    # Tranform data to LDA
    def run_modeling(self, train_data, test_data):

        # Vectorize data
        train_data_vec = TaggedDocument(words=train_data)
        test_data_vec = TaggedDocument(words=test_data)

        # Create LDA model & fit
        self.model = self.create_model()
        # Train model
        self.model.build_vocab(train_data_vec)

        for epoch in range(self.max_epochs):
            print('iteration {0}'.format(epoch))
            self.model.train(train_data_vec,
                        total_examples=self.min_count,
                        epochs=self.max_epochs)
            # decrease the learning rate
            self.alpha -= 0.0002
            # fix the learning rate, no decay
            self.min_alpha = self.alpha

        return train_lda_data, test_lda_data

    # MODEL INFO
    def print_topics(self):
        for idx, topic in self.lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))