from src.fake_news_detector.core.classification import doc2vec_classification as dc    
import pandas as pd 
import numpy as np 

from src.utils import io


if __name__ == "__main__":


    # 1. Read 
    articles = io.read_json_file('/home/elenaruiz/Documents/TFG/FNC/src/data/dataset_content.json')
    df = pd.DataFrame(data=articles['articles'])

    # 2. Create dataframe
    corpus = pd.DataFrame()
    corpus['corpus'] = df['title_subject']
    corpus['fake'] = df['id']
    corpus['id'] = list(range(0,len(df.rows)))

    # 3. DOC2VEC Test
    models = generate_doc2vec_model(data)
    models.get_similarty_doc2vec(data)
    # 4. Check results
    error_1_1 = 0
    error_1_2 = 0
    error_2_1 = 0
    error_2_2 = 0
    for i, row in data.iterrows():
        label = row['label'] * 1
        # Model 1 : MAX
        if row['result_m1_max'] != label:
            error_1_1 += 1
        # Model 1: MEAN
        if row['result_m1_mean'] != label:
            error_1_2 += 1
                # Model 1 : MAX
        if row['result_m1_max'] != label:
            error_1_1 += 1
        # Model 1: MEAN
        if row['result_m1_mean'] != label:
            error_1_2 += 1

    size = len(df.rows)
    print('SIMILARITY TITLE SUBJECT WITH DOC2VEC')
    print('Model 1: Max Top 3:')
    print('%error:', error_1_1*100/size)
    correct_1_1 = size - error_1_1
    print('%correct:', correct_1_1*100/size)
    print('-------------------------------------')
    print('Model 1: Mean Top 3:')
    print('%error:', error_1_2*100/size)
    correct_1_2 = size - error_1_2
    print('%correct:', correct_1_2*100/size)
    print('-------------------------------------')
    print('Model 2: Max Top 3:')
    print('%error:', error_2_1*100/size)
    correct_2_1 = size - error_2_1
    print('%correct:', correct_2_1*100/size)
    print('-------------------------------------')
    print('Model 2: Mean Top 3:')
    print('%error:', error_2_2*100/size)
    correct_2_2 = size - error_2_2
    print('%correct:', correct_2_2*100/size)
