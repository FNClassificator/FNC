
import pandas as pd
from src.utils import io

from src.fake_news_detector.classification import sub_classifications as sc

title_dataset = io.read_json_file('/home/elenaruiz/Documents/TFG/FNC/src/data/dataset_title.json')
similarity_dataset = io.read_json_file('/home/elenaruiz/Documents/TFG/FNC/src/data/dataset_similarity.json')
text_dataset = io.read_json_file('/home/elenaruiz/Documents/TFG/FNC/src/data/dataset.json')

# Datasets
df_title = pd.DataFrame(data=title_dataset['articles'])
df_similarity = pd.DataFrame(data=similarity_dataset['articles'])
df_text = pd.DataFrame(data=text_dataset['articles'])

# Predict TITLE
y_title = sc.get_title_prediction(df_title, 'LR')
print(y_title)

# Predict SIMILARITY
y_similarity = sc.get_similarity_prediction(df_similarity, 'LR')
print(y_similarity)

# Predict TEXT
y_text = sc.get_text_prediction(df_text, 'LR')
print(y_text)

# Join results in a dataset
final_dataset  = []
final_dataset.append(y_title)
final_dataset.append(y_similarity)
final_dataset.append(y_text)
df_final = pd.DataFrame(data=final_dataset)

# Train with some techniques
