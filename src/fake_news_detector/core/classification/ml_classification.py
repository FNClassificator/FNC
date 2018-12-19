
from src.fake_news_detector.core.classification.ml_models import ClassificationModel

""" Splits datasets for training """

def split_dataset_xy(X, y, normalize):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Normalize
    if normalize:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def split_dataset_x(X, normalize):
    X_train, X_test = train_test_split(X, random_state=0)
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
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = ClassificationModel(model_type)
    model.train(X_train, y_train)
    model.test(X_test, y_test)

    if output:
        print('Accuracy of ' + model_type + ' classifier on training set: {:.2f}'
            .format(model.score(X_train, y_train)))
        print('Accuracy of ' + model_type + ' classifier on test set: {:.2f}'
            .format(model.score(X_test, y_test)))
    return model.predict_all(X)