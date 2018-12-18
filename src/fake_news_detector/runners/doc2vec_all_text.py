

import pandas as pd
from src.utils import io
from src.fake_news_detector.core.nlp import clean_text as ct
from sklearn.model_selection import train_test_split
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def run():
    #1. Read dataset (tmp.json)
    articles = io.read_json_file('/home/elenaruiz/Documents/TFG/FNC/src/data/tmp.json')
    df = pd.DataFrame(data=articles['articles'])


    #2. Clean data
    df['corpus'] = df['title']
    i = 0
    for _, row in df.iterrows():
        x = ct.clean_text_by_word(row['title'], True)
        y = ct.clean_text_by_word(row['subtitle'], True)
        z = []
        for sent in row['text']:
            z += ct.clean_text_by_word(sent, True)
        row['corpus'].replace(x + y + z)
        i = i + 1


    #3. Split data
    X_train, X_test = train_test_split(df, random_state=0)
    print(X_train)
    print(X_test)

    #4. Tag each doc
    tagged_data = []
    for i in vector_id:
        tagged_data.append(TaggedDocument(corpus[i],str(df['fake'][i])))

    #3. Doc2Vec Model + build vocab
    max_epochs = 100
    vec_size = 20
    alpha = 0.025
    model = Doc2Vec(alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)
    
    model.build_vocab(tagged_data)
    #4. Test resutls
    for epoch in range(max_epochs):
        #print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    # 5. Test
        
    print(similar_doc)
    i = similar_doc[0][0]
    print(i)
    row = df.loc[[1]]
    print(row['title'], row['text'])
    row = df.loc[[8]]
    print(row['title'], row['text'])
    return

if __name__ == "__main__":
    run()