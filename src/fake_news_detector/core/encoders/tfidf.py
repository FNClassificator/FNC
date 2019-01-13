from gensim.corpora import Dictionary
from gensim import models

##############################
###### TF-IDF ENCODING #######
##############################


class Tfidf_model():

    def __init__(self, filter_by_freq, no_above=0.6, no_below=0.6):
        self.filter_by_freq = filter_by_freq
        self.no_above=no_above
        self.no_below=no_below
        
        self.dictionary = None
        self.tfidf_model = None

    # TF-IDF Model
    def create_dictionary(self, train_data):
        dictionary = Dictionary(train_data)
        if self.filter_by_freq:
            dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)
        return dictionary

    # Transformations
    def data2bow(self, dictionary, data):
        result = [dictionary.doc2bow(doc) for doc in data]
        return result

    def bow2tfidf(self, tfidf_model, data):
        return tfidf_model[data]

    # Getters
    def get_model(self):
        return self.tfidf_model
    
    def get_dictionary(self):
        return self.dictionary


    # USAGE
    # Create TF-IDF Model
    # Tranform data to tfidf
    def run_transformation(self, train_data, test_data):

        self.dictionary = self.create_dictionary(train_data)
        # TRAIN DATA
        train_bow_data  = self.data2bow(self.dictionary,train_data)
        self.tfidf_model = models.TfidfModel(train_bow_data)
        train_tfidf_data = self.bow2tfidf(self.tfidf_model, train_bow_data)

        # TEST DATA
        test_bow_data = self.data2bow(self.dictionary, test_data)
        test_tfidf_data = self.bow2tfidf(self.tfidf_model, test_bow_data)

        return train_tfidf_data, test_tfidf_data

    def print_size(self):
        print("Found {} words.".format(len(self.dictionary.values())))
