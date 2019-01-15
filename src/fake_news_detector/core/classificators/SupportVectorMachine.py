
from sklearn.svm import SVC
from src.fake_news_detector.core.classificators import helpers
from sklearn.model_selection import GridSearchCV

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

def svc_param_selection(X, y, nfolds, kernel):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

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

def run_optimals_models(x_train, y_train, x_test, y_test, output):
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    score_models = {}
    for kernel in kernels:
        # Get optimal params
        params = svc_param_selection(x_train, y_train, 2, kernel)
        svm_model = SVC(kernel=kernel, C= params['C'], gamma=params['gamma'])
        score_models[kernel] = {}
        score_models[kernel]['train'],  score_models[kernel]['test'] = run_model(svm_model, x_train, y_train, x_test, y_test, False)
    # Get top score and model
    # Pos 0: Kernel
    # Pos 1: Score
    max_validation = [None, 0]
    max_training = [None, 0]

    for kernel in score_models:
        if output:
            print('For model', kernel)
            print('Training score: {}. Test score: {}'.format(score_models[kernel]['train'],score_models[kernel]['test']))
        # Check training
        if score_models[kernel]['train'] > max_training[1]:
            max_training[0] = kernel
            max_training[1] = score_models[kernel]['train']
        # Check validation
        if score_models[kernel]['test'] > max_validation[1]:
            max_validation[0] = kernel
            max_validation[1] = score_models[kernel]['test']
    return max_training + max_validation

