
from src.fake_news_detector.core.classification.ml.models import ClassificationModel
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

class BOWModels():

    def __init__(self, train_data, test_data, label_x, label_y):
        # DATASET
        self.train_data = train_data
        self.test_data = test_data
        self.label_x = label_x
        self.label_y = label_y
        # TRANSFORMATION
        self.vectorizers = [
            CountVectorizer( analyzer="word", tokenizer=self.simple, preprocessor=lambda x: x, stop_words="english", max_features=3000), #BOW
            CountVectorizer( analyzer="word", ngram_range=([2,5]), tokenizer=self.simple, preprocessor=lambda x: x, max_features=3000), #N-gram
            TfidfVectorizer( min_df=2, tokenizer=self.simple, preprocessor=None)
        ]
        # MODELS
        self.data_models = []
        self.ml_models = {}

    def simple(self, doc):
        return doc.split(' ')

    def build_models(self):
        for vectorizer in self.vectorizers:
            self.data_models.append(vectorizer.fit_transform(self.train_data[self.label_x].values))

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
            for name in list_classificator_models:
                # Create model
                model = ClassificationModel(name)
                # data: vectorized text \ train_data_ get label
                score = model.train(data, self.train_data[self.label_y].values)
                # Store model
                self.ml_models[name].append(model)
                scores.append('Model Classificator ' + name + '  Score: ' + str(score))
            i += 1
        
        if output:
            for line in scores:
                print(line)
        
        return scores
    
    def get_feature_names(self, output):
        res = []
        for model in self.vectorizers:
            res.append('NEW MODEL')
            res.append(model.get_feature_names())
        
        if output:
            for line in res:
                print(line)
        return res

    def predict_all(self, output):
        target = self.test_data[self.label_y]
        for i in range(0, len(self.vectorizers)):
            # Predict for each ML model
            print('MODEL ', str(i), ':')
            for key in self.ml_models.keys():
                print('Classificator: ', key)
                accuracy, cm = self.predict_one(self.vectorizers[i], self.ml_models[key][i], target, output)       
    
    def predict_one(self, count_vectorizer, classificator, target, output):
        data_features = count_vectorizer.transform(self.test_data['corpus'].values)
        prediction =  classificator.predict_all(data_features)
        return classificator.evaluate_prediction(prediction, target, output)



