
from sklearn.linear_model import LogisticRegression


def create_model():
    return LogisticRegression()

def train_model(model, X_train, y_train):
    return model.fit(X_train,y_train)

def predict(model, X_test):
    return model.predict(X_test)
