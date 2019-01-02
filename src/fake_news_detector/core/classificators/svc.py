
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix  

#  SVC

# kernel: linear / poly / rbf / sigmoid / precomputed Default: rbf
# degree if poly kernel
# gamma: coef Default: 'auto' (1/n_features). 'scale': 1/(n_features*X.std())


#####################
# CREATE MODEL
#####################

def SVC_default():
    return SVC()

# linear / poly / rbf / sigmoid
def SVC_kernel(kernel, degree=1):
    if kernel == 'poly':
        return SVC(kernel=kernel, degree = degree)
    return SVC(kernel=kernel)

def SVC_kernel_gamma(kernel, gamma, degree=1)
    if kernel == 'poly':
        return SVC(kernel=kernel, gamma=gamma, degree = degree)
    return SVC(kernel=kernel, gamma=gamma)

#####################
# TRAIN
#####################
def train(model, x_train, y_train):
    return model.train(x_train, y_train)

#####################
# PREDICT
#####################

def precit_all(model, x_test):
    return model.predict(x_test)

#####################
# EVALUATION
#####################

def get_evaluation(y_test, y_pred):
    confusion_m = confusion_matrix(y_test,y_pred)
    class_report = classification_report(y_test,y_pred)
    return confusion_m, class_report

def print_evaluation(y_test, y_pred):

    confusion_m, class_report = get_evaluation(y_test, y_pred)

    print('Confusion matrix:')
    print(confusion_m)
    print('REPORT:')
    print(class_report)
    return class_report


#####################
# ALL 
#####################
def run_results(model, x_train, y_train, x_test, y_test):
    train(model, x_train, y_train)
    y_pred = predict_all(x_test)
    return print_evaluation(y_test, y_pred)
    