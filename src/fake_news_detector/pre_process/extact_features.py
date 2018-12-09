from src.fake_news_detector.nlp import clean_text as ct
from src.fake_news_detector.nlp import quantity as q
from src.fake_news_detector.nlp import similarity as s
from src.fake_news_detector.nlp import sentiment as sent
from src.fake_news_detector.nlp import tokenize as tk
from src.utils import io
import itertools


def get_title_info(dataset):
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
    return dataset_all


def get_similarity_info(dataset):
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
        dict_t['similarity_text_title'] = s.get_jaccard_similarity(
            token_text, token_title)
        if row['subtitle'] == '':
            dict_t['similarity_title_subtitle'] = 0.5
            dict_t['similarity_text_subtitle'] = 0.5
        else:
            token_subtitle = ct.clean_text_words(row['subtitle'])
            dict_t['similarity_title_subtitle'] = s.get_jaccard_similarity(
                token_subtitle, token_title)
            dict_t['similarity_text_subtitle'] = s.get_jaccard_similarity(
                token_text, token_subtitle)
        # Tokens
        if row['fake']:
            dict_t['fake'] = 1
        else:
            dict_t['fake'] = 0
        dataset_all['articles'].append(dict_t)
    io.write_json_file('src/data/dataset_similarity.json', dataset_all)
    return dataset_all


def get_text_info(dataset):
    dataset_all = {
        'articles': []
    }
    for _, row in dataset.iterrows():
        dict_t = {
            'text_length': None,
            'text_sentences': None,
            'text_adj_words': None,
            'text_verbs_words': None,
            'text_modal_verbs': None,
            #'similarity_betweent_paragraphs': None,
            'sentiment': None,
            'fake': None
        }

        # Variables
        paragraphs = row['text']
        all_text = []
        all_sentences = []

        for paragraph in paragraphs:
            sentences = tk.tokenize_by_sentences(paragraph)
            all_sentences += sentences
            text = ct.clean_text_words(paragraph)
            all_text += text

        tag_text = q.get_tags(all_text)
        # Features
        if row['fake']:
            dict_t['fake'] = 1
        else:
            dict_t['fake'] = 0
        dict_t['text_length'] = q.n_words(all_text)
        dict_t['text_sentences'] = q.n_sentences(all_sentences)
        dict_t['text_adj_words'] = q.perct_adj_words(tag_text)
        dict_t['text_verbs_words'] = q.perct_verb_words(tag_text)
        dict_t['text_modal_verbs'] = q.pert_modal_verbs(all_text)

        # SIMILARITY
        #documents = row['text']
        #dict_t['similarity_betweent_paragraphs'] = s.get_documents_similarities(documents)

        # SENTIMENT
        sentiment_sum = 0
        for sentence in all_sentences:
            sentiment_sum += sent.get_sentiment_by_phrases(sentence)
        dict_t['sentiment'] = sentiment_sum / len(all_sentences)
        dataset_all['articles'].append(dict_t)
    io.write_json_file('src/data/dataset_text.json', dataset_all)
    return dataset_all
