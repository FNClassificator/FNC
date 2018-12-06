from src.fake_news_detector.helpers.process_data import pre_process
from src.fake_news_detector.helpers.nlp import feature_extractions as fe
from src.fake_news_detector.helpers.nlp import clean_text as ct
from src.fake_news_detector.helpers.nlp import quantity as q
from src.utils import io 

# OBJECTIVE: Classificate by title

# 1. Get dataset
dataset = pre_process.modelate_dataset()

# 2. Clean title
dict_t = {
    'title_token_clean': None,
    'title_token_clean_tag': None,
    'title_length': None,
    'title_noun_words': None,
    'title_adj_words': None,
    'title_verbs_words': None
}

dataset_all = {
    'articles': []
}

for _, row in dataset.iterrows():

    # Tokens
    dict_t['title_fake'] = row['fake']
    token_t = ct.clean_text_words(row['title'])
    dict_t['title_token_clean'] = token_t
    # Tags
    tag_text = q.get_tags(token_t)
    dict_t['title_token_clean_tag'] = tag_text
    # Features
    dict_t['title_length'] = q.n_words(tag_text)
    dict_t['title_noun_words'] = q.perct_noun_words(tag_text)
    dict_t['title_adj_words'] = q.perct_adj_words(tag_text)
    dict_t['title_verbs_words'] = q.perct_verb_words(tag_text)
    dataset_all['articles'].append(dict_t)

io.write_json_file('src/data/out.json', dataset_all)