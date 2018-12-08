from src.fake_news_detector.helpers.process_data import pre_process
from src.fake_news_detector.helpers.nlp import clean_text as ct
from src.fake_news_detector.helpers.nlp import quantity as q
from src.fake_news_detector.helpers.nlp import similarity as s
from src.fake_news_detector.helpers.nlp import sentiment as sent
from src.utils import io 
import itertools

# OBJECTIVE: Classificate by title

def compute_title(dataset):
    dataset_all = {
        'articles': []
    }
    for _, row in dataset.iterrows():
        dict_t = {
            'title_length': None,
            'title_adj_words': None,
            'title_verbs_words': None,
            'title_modal_verbs': None,
            'fake': None,
            'sentiment': None
        }

        # Tokens
        if row['fake']:
            dict_t['fake'] = 1
        else: 
            dict_t['fake'] = 0
        # Tags
        token_text = ct.clean_text_words(row['title'])
        tag_text = q.get_tags(token_text)
        # Features
        dict_t['title_length'] = q.n_words(tag_text)
        dict_t['title_adj_words'] = q.perct_adj_words(tag_text)
        dict_t['title_verbs_words'] = q.perct_verb_words(tag_text)
        dict_t['title_modal_verbs'] = q.pert_modal_verbs(token_text)

        # SENTIMENT 
        sentiment = sent.get_sentiment_by_phrases(row['title'])
        dict_t['sentiment'] = sentiment 
        dataset_all['articles'].append(dict_t)

    io.write_json_file('src/data/dataset_title.json', dataset_all)

def compute_similarity(dataset):
    dataset_all = {
        'articles': []
    }
    for _, row in dataset.iterrows():
        dict_t = {
            'similarity_text_title': None,
            'similarity_text_subtitle': None,
            'similarity_title_subtitle': None,
            'fake': None
        }
        token_text = ct.clean_text_words(row['title'])
        token_title = ct.clean_text_words(' '.join(row['text']))
        dict_t['similarity_text_title'] = s.get_jaccard_similarity(token_text,token_title)
        if row['subtitle'] == '':
            dict_t['similarity_title_subtitle'] = 0.5
            dict_t['similarity_text_subtitle'] = 0.5
        else:
            token_subtitle = ct.clean_text_words(row['subtitle'])
            dict_t['similarity_title_subtitle'] = s.get_jaccard_similarity(token_subtitle,token_title)
            dict_t['similarity_text_subtitle'] = s.get_jaccard_similarity(token_text,token_subtitle)
        # Tokens
        if row['fake']:
            dict_t['fake'] = 1
        else: 
            dict_t['fake'] = 0
        dataset_all['articles'].append(dict_t)
    io.write_json_file('src/data/dataset_similarity.json', dataset_all)
        

def compute_all():
    # 1. Get dataset
    dataset = pre_process.modelate_dataset()
    # 2. Clean title and extract features
    compute_title(dataset)
    compute_similarity(dataset)

if __name__ == '__main__':
    compute_all()