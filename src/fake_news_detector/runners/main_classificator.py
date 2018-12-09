
import pandas as pd
from src.utils import io

from src.fake_news_detector.classification import sub_classifications as sc

title_dataset = io.read_json_file('/home/elenaruiz/Documents/TFG/FNC/src/data/dataset_title.json')
similarity_dataset = io.read_json_file('/home/elenaruiz/Documents/TFG/FNC/src/data/dataset_similarity.json')
text_dataset = io.read_json_file('/home/elenaruiz/Documents/TFG/FNC/src/data/dataset_text.json')

# Datasets
df_title = pd.DataFrame(data=title_dataset['articles'])
df_similarity = pd.DataFrame(data=similarity_dataset['articles'])
df_text = pd.DataFrame(data=text_dataset['articles'])

# Predict TITLE
y_title = sc.get_title_prediction(df_title, 'LR', False)
print(y_title)

# Predict SIMILARITY
y_similarity = sc.get_similarity_prediction(df_similarity, 'LR', False)
print(y_similarity)

# Predict TEXT
y_text = sc.get_text_prediction(df_text, 'LR', False)
print(y_text)

# Join results in a dataset
data_final = {
    'articles': []
}
for i in range(0,len(y_title)):
    aux = { }
    aux['title'] = y_title[i]
    aux['similarity'] = y_similarity[i]
    aux['text'] = y_text[i]
    aux['fake'] = title_dataset['articles'][i]['fake']
    data_final['articles'].append(aux)

df_final = pd.DataFrame(data=data_final['articles'])


# Train with some techniques
model_type = 'LR'
pred = sc.get_main_prediction(df_final, model_type, True)


model_type = 'DTC'
pred = sc.get_main_prediction(df_final, model_type, True)


model_type = 'KNC'
pred = sc.get_main_prediction(df_final, model_type, True)


model_type = 'LDA'
pred = sc.get_main_prediction(df_final, model_type, True)


model_type = 'GNB'
pred = sc.get_main_prediction(df_final, model_type, True)


model_type = 'SVC'
pred = sc.get_main_prediction(df_final, model_type, True)
