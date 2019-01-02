
from sklearn.svm import SVC



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

def SVC_kernel_gamma(kernel, gamma, degree=1):
    if kernel == 'poly':
        return SVC(kernel=kernel, gamma=gamma, degree = degree)
    return SVC(kernel=kernel, gamma=gamma)

#####################
# TRAIN
#####################
def train(model, x_train, y_train):
    return model.fit(x_train, y_train)

#####################
# PREDICT
#####################

def precit_all(model, x_test):
    return model.predict(x_test)




#####################
# ALL 
#####################
def run_results(model, x_train, y_train, x_test, y_test):
    train(model, x_train, y_train)
    y_pred = precit_all(model, x_test)
    return print_evaluation(model, x_train, y_train, y_test, y_pred)
    