from src.fake_news_detector.helpers.process_data import pre_process
from src.fake_news_detector.helpers.nlp import feature_extractions as fe

import numpy as np
from sklearn.lda import LDA
# OBJECTIVE: Classificate by title

# 1. Get dataset
dataset = pre_process.modelate_dataset()

# 2. Clean title
pre_process.tokenize_by_word_and_clean(dataset, 'title')
print(dataset["title_token_clean"][0])
print(dataset["title"][0])

# 3. Extract informations
dataset['title_perct_adj'] = dataset['title_token_clean'] # Copy of column
dataset['title_perct_noun'] = dataset['title_token_clean'] # Copy of column
dataset['title_subjectivity'] =  dataset['title_token_clean']
dataset['title_polarity'] =  dataset['title_token_clean']
for _, row in dataset.iterrows():
    # Extract % adjectives
    row['title_perct_adj'] = fe.count_adjectives(row['title_token_clean'])
    # Extract % nouns
    row['title_perct_noun'] = fe.count_common_nouns(row['title_token_clean'])
    # Sentiment 
    sentiment = fe.get_sentiment(row['title'])
    # Subjectivity
    row['title_subjectivity'] = sentiment.subjectivity
    # Polarity
    row['title_polarity'] = sentiment.polarity

print(dataset["title_perct_adj"][0])
print(dataset["title_perct_noun"][0])
print(dataset["title_subjectivity"][0])
print(dataset["title_polarity"][0])

# 3. Normalize


# 4. Split in training and validation
X = dataset[['title_perct_adj','title_perct_noun','title_subjectivity','title_polarity']]
y = dataset['result']
# 4. Create LDA classificator
clf = LDA()
clf.fit(X, y)
# 5. Predict


# 6. Get results
