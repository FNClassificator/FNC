
from src.fake_news_detector.core.classification.ml import models as mml

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

class BOWModels():

    def __init__(self, train_data, test_data, label_x, label_y):
        # DATASET
        self.train_data = train_data
        self.test_data = test_data
        self.label_x = label_x
        self.label_y = label_y
        # TRANSFORMATION
        self.vectorizers = [
            CountVectorizer( analyzer="word", tokenizer=None, preprocessor=None, max_features=3000), #BOW
            CountVectorizer( analyzer="char", ngram_range=([2,5]), tokenizer=None, preprocessor=None, max_features=3000), #N-gram
            TfidfVectorizer( min_df=2, tokenizer=nltk.word_tokenize, preprocessor=None, stop_words='english')
        ]
        # MODELS
        self.data_models = []
        self.ml_models = {}

    def build_models(self, label_x):
        for vectorizer in self.vectorizers:
            self.data_models.append(vectorizer.fit_transform(self.train_data[label_x]))

    """
    With all list_classificator_models from ClassificatorModel() object
     - Train with them
     - Store models in class 
     RETURN: Returns all training score for each model of encoding and classificator model
    """
    def train_with_classificators(self, list_classificator_models, output):
        scores = []
        # Init dictionaries
        for name in list_classificator_models:
            self.ml_models[name] = []

        # Train and store models
        i = 0
        for data in self.data_models:
            scores.append('MODEL VECTORIZER ' + str(i) + ':')
            for name in list_classficator_models:
                # Create model
                model = mml.ClassifcationModel(name)
                # data: vectorized text \ train_data_ get label
                score = model.train(data, self.train_data[self.label_y])
                # Store model
                self.ml_models[name].append(model)
                scores.append('Model Classificator ' + name + '  Score: ' + str(score))
        
        if output:
            for line in score:
                print(line)
        
        return scores
    
    
    def predict_all(self, output):
        target = self.test_data[self.label_y]
        for i in range(0, len(self.data_models)):
            # Predict for each ML model
            for classificator in self.ml_models:
                accuracy, cm = self.predict_one(self.data_models[i], classificator, target, output)
                
    
    def predict_one(self, count_vectorizer, classificator, target, output):
        data_features = count_vectorizer.transform(self.training_data)
        prediction =  classificator.predict(data_features)
        return classificator.evaluate_prediction(prediction, target, output)



