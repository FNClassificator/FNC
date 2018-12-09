from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


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
