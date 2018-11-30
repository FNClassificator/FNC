from src.fake_news_detector.helpers.process_data import pre_process
from src.fake_news_detector.helpers.nlp import feature_extractions as fe

# OBJECTIVE: Classificate by title

# 1. Get dataset
dataset = pre_process.modelate_dataset()

# 2. Clean title
pre_process.tokenize_by_word_and_clean(dataset, 'title')
print(dataset["title_token_clean"][0])
print(dataset["title"][0])


dataset['title_perct_adj'] = dataset['title_token_clean'] # Copy of column
dataset['title_perct_noun'] = dataset['title_token_clean'] # Copy of column
for _, row in dataset.iterrows():
    # Extract % adjectives
    row['title_perct_adj'] = fe.count_adjectives(row['title_perct_adj'])
    # Extract % nouns
    row['title_perct_noun'] = fe.count_uncommon_nouns(row['title_perct_noun'])


# 3. Vectorize


# 4. Create LDA classificator


# 5. Predict


# 6. Get results
