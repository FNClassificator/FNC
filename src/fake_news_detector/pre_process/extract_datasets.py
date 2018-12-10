from src.fake_news_detector.nlp import tokenize as tk
from src.fake_news_detector.nlp import clean_text as ct

from src.fake_news_detector.nlp.features import sentiment as sent
from src.fake_news_detector.nlp.features import words as w
from src.fake_news_detector.nlp.features import quantity as q
from src.fake_news_detector.nlp import similarity as s


from src.utils import io
import itertools

def get_all_text_tokenized(row):
    # Title
    title_token_word = ct.clean_text_words(row['title'])
    # Subtitle
    if row['subtitle'] == '':
        subtitle_token_word = []
        subtitle_token_sent = []
    else:
        subtitle_token_word = ct.clean_text_words(row['subtitle'])
        subtitle_token_sent = ct.clean_text_sent(row['subtitle'])
    # Text
    text_token_paragraph_word = []
    for text in row['text']:
        text_token = ct.clean_text_words(row['text'])
        text_token_paragraph_word.append(text_token)
    joined_text = '-'.join(row['text'])

    text_token_word = ct.clean_text_words(joined_text)
    text_token_sent = ct.clean_text_words(joined_text)

    # Title + Subtitle + Text
    all_text = row['title'] + ' ' + row['subtitle'] + ' ' + joined_text
    all_token_word = ct.clean_text_words(all_text)
    all_token_sent = ct.clean_text_words(all_text)

    tokendata = {
        'title': {
            'word': title_token_word
        },
        'subtitle': {
            'word': subtitle_token_word,
            'sent': subtitle_token_sent
        }
        'text': {
            'word': text_token_word,
            'sent': text_token_sent,
            'paragraph': text_token_paragraph_word,
            'joinend_raw': joined_text
        },
        'all': {
            'word': all_token_word,
            'sent': all_token_sent
        }
    }
    return tokendata

# DATASET 1: Content
# Objective: Get info about article's words
# Variables :
#   - Positive words 
#   - Negative words
#   - Common noun words
#   - Adjective words
#   - Conjunction words
#   - Noun phrases words (N-grams)
#   And the same, but only for headlines
def get_content_dataset(dataset):
    content_dataset = {
        'articles': []
    }
    for _, row in dataset.iterrows():
        dict_t = {
            'positive_words': None,
            'negative_words': None,
            'common_noun_words': None,
            'adjective_words': None,
            'conjunction_words': None,
            'noun_phrases_words': None,
            'title_positive_words': None,
            'title_negative_words': None,
            'title_common_noun_words': None,
            'title_adjective_words': None,
            'title_conjunction_words': None,
            'title_noun_phrases_words': None,
            'fake': None
        }
        
        tokendata = get_all_text_tokenized(row)

        dict_t['positive_words'] = sent.get_positive_words(tokendata['all']['word'])
        dict_t['negative_words'] = sent.get_negative_words(tokendata['all']['word'])

        tagged_words = q.get_tags(tokendata['all']['word'])
        dict_t['common_noun_words'] = w.get_common_nouns(tagged_words)
        dict_t['adjective_words'] = w.get_adj_words(tagged_words)
        dict_t['conjunction_words'] = w.get_conj_words(tagged_words)
        dict_t['noun_phrases_words'] = w.get_noun_phrases_words(tagged_words)
    
        # About title
        dict_t['title_positive_words'] = sent.get_positive_words(tokendata['title']['word'])
        dict_t['title_positive_words'] = sent.get_negative_words(tokendata['title']['word'])

        tagged_words = q.get_tags(tokendata['title']['word'])
        dict_t['title_common_noun_words'] = w.get_common_nouns(tagged_words)
        dict_t['title_adjective_words'] = w.get_adj_words(tagged_words)
        dict_t['title_conjunction_words'] = w.get_conj_words(tagged_words)
        dict_t['title_noun_phrases_words'] = w.get_noun_phrases_words(tagged_words)
        dict_t['fake'] = row['fake']
        content_dataset.append(dict_t)
    return content_dataset

# DATASET 2: Style
# Objective: Get info about how is written
# Variables :
#   - Sentiment
#   - # of words
#   - # of sentences
#   - % of verbs
#   - % of nouns
#   - % of adj
#   - % of conj and prepositions
#   - % of positive words
#   - % of negative words
#   - % of different words
#   - # of quotes
#   - mean of words per sentence
#   - mean characters per word
#   - mean noun phrases
#   - reduncancy
#   And the same in some cases for only for headlines
def get_style_dataset(dataset):
    content_dataset = {
        'articles': []
    }
    for _, row in dataset.iterrows():
        dict_t = {
            'sentiment': None,
            'n_words': None,
            'n_sentences': None,
            'pert_total_verbs': None,
            'pert_total_nouns': None,
            'pert_total_adj': None,
            'pert_total_conj_prep': None,
            'pert_total_positive_words': None,
            'pert_total_negative_words': None,
            'pert_different_words': None,
            'n_quotes': None,
            'mean_words_per_sentence': None,
            'mean_character_per_word': None,
            'mean_noun_phrases': None,
            'redundancy': None,
            'title_sentiment': None,
            'title_n_words': None,
            'title_n_sentences': None,
            'title_pert_total_conj_prep': None,
            'title_pert_total_positive_words': None,
            'title_pert_total_negative_words': None,
            'title_mean_noun_phrases': None,
            'fake': None
        }
        tokendata = get_all_text_tokenized(row)

        dict_t['sentiment'] = sent.get_sentiment(tokendata['all']['word'])
        dict_t['n_words'] = q.n_words(tokendata['all']['word'])
        dict_t['n_sentences'] = q.n_sentences(tokendata['all']['word'])
        
        tagged_words = q.get_tags(tokendata['all']['word'])
        dict_t['pert_total_verbs'] = q.perct_verb_words(tagged_words)
        dict_t['pert_total_nouns'] = q.perct_noun_words(tagged_words)
        dict_t['pert_total_adj'] = q.perct_adj_words(tagged_words)
        dict_t['pert_total_conj_prep'] = q.perct_conj_words(tagged_words)
        dict_t['pert_total_positive_words'] = q.n_sentences(tokendata['all']['word'])
        dict_t['pert_total_negative_words'] = q.n_sentences(tokendata['all']['word'])
        dict_t['pert_different_words'] = q.n_sentences(tokendata['all']['word'])
        dict_t['n_quotes'] = q.n_sentences(tokendata['all']['word'])
        dict_t['mean_words_per_sentence'] = q.n_sentences(tokendata['all']['word'])
        dict_t['mean_character_per_word'] = q.n_sentences(tokendata['all']['word'])
        dict_t['mean_noun_phrases'] = q.n_sentences(tokendata['all']['word'])
        style_dataset.append(dict_t)
    return style_dataset

# DATASET 3: Similarity
# Objective: Get info about diferences inside it content and the other documents
# Variables :
#   - Similarity between title and subtitle
#   - Similarity between title and text
#   - Similarity between text and subtitle
#   - Similarity between paragraphs in text (mean, min, max)
#   - Similarity between title of others
#   - Similarity between other texts
def get_content_dataset(dataset):
    dataset_all = {
        'articles': []
    }
    for _, row in dataset.iterrows():
        dict_t = {
            'positive_words': None,
            'negative_words': None,
            'common_noun_words': None,
            'adjective_words': None,
            'conjunction_words': None,
            'noun_phrases_words': None,
            'title_positive_words': None,
            'title_negative_words': None,
            'title_common_noun_words': None,
            'title_adjective_words': None,
            'title_conjunction_words': None,
            'title_noun_phrases_words': None,
            'fake': None
        }
    return dataset_all


# Get dataset about title info
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
