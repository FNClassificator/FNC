from sklearn.metrics import classification_report, confusion_matrix, precision_score

#####################
# EVALUATION
#####################

def get_evaluation(y_test, y_pred):
    confusion_m = confusion_matrix(y_test,y_pred)
    class_report = classification_report(y_test,y_pred)
    score = precision_score(y_test, y_pred)
    return confusion_m, class_report, score

def print_evaluation(model, x_train, y_train, y_test, y_pred):

    confusion_m, class_report, score = get_evaluation(y_test, y_pred)

    print('Confusion matrix:')
    print(confusion_m)
    print('REPORT:')
    print(class_report)
    print('Train precision score:', model.score(x_train, y_train))
    print('Test precision score:', score)
    return class_report