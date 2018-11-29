from src.fake_news_detector.helpers.read_data import dataframe
from src.fake_news_detector.helpers.read_data import io
from src.fake_news_detector.helpers.nlp import clean_text

from src.fake_news_detector.helpers.nlp import  clean_text

def modelate_dataset():
    path = 'src/fake_news_detector/data/tmp.py'
    # Read file
    io.read_json_file(path)
    # Create dataframe
    return dataframe.get_dataframe_from_json()

# name: original column
def tokenize_by_word_and_clean(dataset, name):
    new_name = name + '_sent'
    dataset[new_name] = dataset[name] # Copy of column
    for index, row in dataset.iterrow():
        row[new_name] = clean_text.clean_text_words(row[new_name])