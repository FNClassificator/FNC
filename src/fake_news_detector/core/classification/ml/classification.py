
from src.fake_news_detector.core.classification.ml.models import ClassificationModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from IPython.display import Markdown, display


def printmd(string):
    display(Markdown(string))

""" Splits datasets for training """

def split_dataset_xy(X, y, normalize):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)
    # Normalize
    if normalize:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def split_dataset_x(X, normalize):
    X_train, X_test = train_test_split(X, random_state=6)
    # Normalize
    if normalize:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test


""" Use of ML classificators """

"""
Build ML and test it results
@df: All dataset
@model_type: type of model we want to use
@list_t: vector of features to train model
@label: feature that will be the output of classification
@outupu: True if we want to print the results
"""
def get_prediction(df, model_type,list_t, label, output):
    y = df[label]
    X = df[list_t]
    X_train, X_test, y_train, y_test = split_dataset_xy(X, y, True)

    model = ClassificationModel(model_type)
    model.train(X_train, y_train)
    model.test(X_test, y_test)

    accuracy_train = model.score(X_train, y_train)
    accuracy_test  = model.score(X_test, y_test)
    if output:
        text_labels = ' '.join(list_t)
        printmd('Accuracy of **' + model_type + '** classifier with **' + str(len(list_t)) + '** variables:')
        printmd('*'+text_labels+'*')
        printmd('**Training set: {:.2f}**'
            .format(accuracy_train))
        printmd('**Test set: {:.2f}**'
            .format(accuracy_test))
        print('\n')
    return model.predict_all(X), accuracy_train, accuracy_test