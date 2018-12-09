import numpy as np
import pandas as pd

from src.utils import io
from src.fake_news_detector.classification.sub_classifications import get_title_prediction

if __name__ == "__main__":

    articles = io.read_json_file('/home/elenaruiz/Documents/TFG/FNC/src/data/dataset_title.json')
    df = pd.DataFrame(data=articles['articles'])


    model_type = 'LR'
    pred = get_title_prediction(df, model_type, True)


    model_type = 'DTC'
    pred = get_title_prediction(df, model_type, True)


    model_type = 'KNC'
    pred = get_title_prediction(df, model_type, True)


    model_type = 'LDA'
    pred = get_title_prediction(df, model_type, True)


    model_type = 'GNB'
    pred = get_title_prediction(df, model_type, True)


    model_type = 'SVC'
    pred = get_title_prediction(df, model_type, True)