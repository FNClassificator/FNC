
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.fake_news_detector.core.classification.models import ClassificationModel


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def get_prediction(df, model_type,list_t, output):
    y = df['fake']
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


def get_similarity_prediction(df, model_type, output):
    X = df[['similarity_text_title','similarity_text_subtitle','similarity_title_subtitle']]
    y = df['fake']
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


def get_text_prediction(df, model_type, output):
    X = df[['text_length','text_sentences','text_adj_words', 'text_verbs_words', 'text_modal_verbs', 'sentiment']]
    y = df['fake']
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


def get_main_prediction(df, model_type, output):
    X = df[['title','similarity','text']]
    y = df['fake']
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