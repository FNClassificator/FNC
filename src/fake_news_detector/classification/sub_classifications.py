
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.fake_news_detector.classification.models import ClassificationModel


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def get_title_prediction(df, model_type):
    X = df[['sentiment','title_adj_words','title_verbs_words']]
    y = df['fake']
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = ClassificationModel(type)
    model.train(X_train, y_train)
    model.test(X_test, y_test)
    return model.predict_all(X)


def get_similarity_prediction(df, model_type):
    X = df[['similarity_text_title','similarity_text_subtitle','similarity_title_subtitle']]
    y = df['fake']
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = ClassificationModel(type)
    model.train(X_train, y_train)
    model.test(X_test, y_test)
    return model.predict_all(X)


def get_text_prediction(df, model_type):
    X = df[['text_length','text_sentences','text_adj_words', 'text_verbs_words', 'text_modal_verbs', 'sentiment']]
    y = df['fake']
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = ClassificationModel(type)
    model.train(X_train, y_train)
    model.test(X_test, y_test)
    return model.predict_all(X)