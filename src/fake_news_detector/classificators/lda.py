import numpy
from sklearn.lda import LDA

# Info about this classificator:
# https://scikit-learn.org/0.15/modules/generated/sklearn.lda.LDA.html
# https://sebastianraschka.com/Articles/2014_python_lda.html#introduction

class LDAClassificator:
    def __init__(self, training_data, validation_data):
        self.lda_clf = LDA()
        self.training_data_x = training_data['x']
        self.training_data_y = training_data['y']
        self.validation_data = validation_data


    def get_classificator(self):
        return self.lda_clf


    def train(self):
        self.lda_clf.train(self.training_data_x, self.training_data_y)
    

    def predict(self):
        return self.lda_predict(self.validation_data)
