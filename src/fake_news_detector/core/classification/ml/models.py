from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def get_model(var):
    switcher = {
        'LR': LogisticRegression(),
        'DTC': DecisionTreeClassifier(),
        'KNC': KNeighborsClassifier(),
        'LDA': LinearDiscriminantAnalysis(),
        'GNB': GaussianNB(),
        'SVC': SVC()
    }
    return switcher.get(var, -1)


class ClassificationModel():

    def __init__(self, type):
        self.model = get_model(type)

    def train(self,X_train,y_train):
        self.model.fit(X_train, y_train)
        score = self.model.score(X_train, y_train)
        return score
    
    def test(self,X_test,y_test):
        score = self.model.score(X_test, y_test)
        return score
    
    def predict_all(self,X):
        y = self.model.predict(X)
        return y

    def show_topics(self, n_topics, n_words):
        return self.model.show_topics(num_topics=n_topics, num_words=n_words)

    def score(self, X, y):
        return self.model.score(X,y)

    def confusion_matrix(self):
        return 
    
    def accuracy_score(self):
        return

    def evaluate_prediction(self, prediction, target, output):
        accuracy = accuracy_score(target, prediction)
        cm = confusion_matrix(target, prediction)

        if output:
            print('Accuracy: ', accuracy)
            print('Confusion matrix: ', cm)
            print('(row=expected, col=predicted)')
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            self.plot_confusion_matrix(cm_normalized, title='Confusion Matrix Normalized')
        return accuracy, cm

    def plot_confusion_matrix(self, cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tags = ['True', 'False']
        tick_marks = np.arange(len(tags))
        target_names = tags
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')