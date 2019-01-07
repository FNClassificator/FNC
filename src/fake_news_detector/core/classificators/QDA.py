from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from src.fake_news_detector.core.classificators import lr, helpers

def run_performance(qda_model, X_train,y_train,X_test,y_test, scaler):
    X_train, X_test = train(qda_model, X_train, X_test, y_train, scaler)  # Modelate
    y_pred = predict(qda_model, X_test) # Classify
    helpers.print_evaluation(qda_model, X_train, y_train, y_test, y_pred)
    #splot = visualize(lda_model, X_test, y_test, y_pred, 1)
    #plot_lda_cov(lda_model, splot)

####################
# MODEL   FUNCTIONS
####################

def create_QDA_model():
    return QDA()

def train(qda_model, X_train,X_test, y_train,scaler):
    if scaler:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    qda_model.fit(X_train, y_train)
    return X_train, X_test

def predict(qda_model, X_test):
    return qda_model.predict(X_test)
