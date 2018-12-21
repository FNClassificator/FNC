from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix

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

    def score(self, X, y):
        return self.model.score(X,y)

    def confusion_matrix(self):
        return 
    
    def accuracy_score(self):
        return

    def evaluate_prediction(self, prediction, target, output):
        accuracy = accuracy_score(target, prediction)
        cm = confusion_matrix(target, predictions)

        if output:
            print('Accuracy: ', accuracy)
            print('Confusion matrix: ', cm)
            print('(row=expected, col=predicted)')
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plot_confusion_matrix(cm_normalized, 'Confusion Matrix Normalized')
        return accuracy, cm