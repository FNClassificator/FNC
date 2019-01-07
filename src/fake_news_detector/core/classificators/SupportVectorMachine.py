
from sklearn.svm import SVC
from src.fake_news_detector.core.classificators import helpers


#  SVC

# kernel: linear / poly / rbf / sigmoid / precomputed Default: rbf
# degree if poly kernel
# gamma: coef Default: 'auto' (1/n_features). 'scale': 1/(n_features*X.std())


#####################
# CREATE MODEL
#####################

# linear / poly / rbf / sigmoid
def create_SVC(kernel, degree=1):
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
def run_model(model, x_train, y_train, x_test, y_test, output):
    train(model, x_train, y_train)
    y_pred = precit_all(model, x_test)

    score_train, score_test = helpers.print_evaluation(model, x_train, y_train, y_test, y_pred, output)
    return score_train, score_test
    
def run_models(models, x_train, y_train, x_test, y_test):
    scores = {}
    for model in models:
        score_train, score_test = run_model(models[model], x_train, y_train, x_test, y_test, False)
        scores[model] = {}
        scores[model]['train'] =  score_train
        scores[model]['test'] = score_test
    return scores