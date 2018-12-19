from src.fake_news_detector.core.nlp import tokenize as tk
from src.fake_news_detector.core.nlp import clean_text as ct

from src.fake_news_detector.core.nlp.features import sentiment as sent
from src.fake_news_detector.core.nlp.features import words as w
from src.fake_news_detector.core.nlp.features import quantity as q
from src.fake_news_detector.core.nlp.features import similarity as s
from src.fake_news_detector.core.nlp.features import subject as sub


from src.utils import io
import itertools

def get_all_text_tokenized(row):
    # Title
    title_token_word = ct.clean_text_by_word(row['title'])
    title_token_sent = ct.clean_text_by_sentence(row['title'])
    # Subtitle
    if row['subtitle'] == '':
        subtitle_token_word = []
        subtitle_token_sent = []
    else:
        subtitle_token_word = ct.clean_text_by_word(row['subtitle'])
        subtitle_token_sent = ct.clean_text_by_sentence(row['subtitle'])
    # Text
    text_token_paragraph_word = []
    for text in row['text']:
        text_token = ct.clean_text_by_sentence(text)
        text_token_paragraph_word.append(text_token[0])
    
    joined_text = '-'.join(row['text'])
    text_token_word = ct.clean_text_by_word(joined_text)
    text_token_sent = ct.clean_text_by_sentence(joined_text)

    # Title + Subtitle + Text
    all_text = row['title'] + ' ' + row['subtitle'] + ' ' + joined_text
    all_token_word = ct.clean_text_by_word(all_text)
    all_token_sent = ct.clean_text_by_sentence(all_text)

    tokendata = {
        'title': {
            'word': title_token_word,
            'sent': title_token_sent
        },
        'subtitle': {
            'word': subtitle_token_word,
            'sent': subtitle_token_sent
        },
        'text': {
            'word': text_token_word,
            'sent': text_token_sent,
            'paragraph': text_token_paragraph_word,
            'joined_raw': joined_text
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
#   TODO: Subjects, subjective words
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
        sentiments = sent.get_words_by_sentiment(tokendata['all']['word'])
        dict_t['positive_words'] = sentiments[0]
        dict_t['negative_words'] = sentiments[1]

        tagged_words = q.get_tags(tokendata['all']['word'])
        dict_t['common_noun_words'] = w.get_unique_type_words(tagged_words,'N')
        dict_t['adjective_words'] = w.get_unique_type_words(tagged_words,'J')
        dict_t['conjunction_words'] = w.get_unique_type_words(tagged_words, 'C')
        dict_t['noun_phrases_words'] = w.get_noun_phrases(tagged_words)
        
    
        # About title
        sentiments = sent.get_words_by_sentiment(tokendata['title']['word'])
        dict_t['title_positive_words'] = sentiments[0]
        dict_t['title_negative_words'] = sentiments[1]

        tagged_words = q.get_tags(tokendata['title']['word'])
        dict_t['title_common_noun_words'] = w.get_unique_type_words(tagged_words, 'N')
        dict_t['title_adjective_words'] = w.get_unique_type_words(tagged_words, 'J')
        dict_t['title_conjunction_words'] = w.get_unique_type_words(tagged_words, 'C')
        dict_t['title_noun_phrases_words'] = w.get_noun_phrases(tagged_words)
        dict_t['title_subject'] = sub.get_subjects(tokendata['title']['sent'])
        dict_t['fake'] = row['fake']
        content_dataset['articles'].append(dict_t)
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
#   TODO: Has generic subject, has tentative words, numer of quotes, redundacy
def get_style_dataset(dataset):
    style_dataset = {
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
            'title_sentiment': None,
            'title_n_words': None,
            'title_pert_total_conj_prep': None,
            'title_pert_total_positive_words': None,
            'title_pert_total_negative_words': None,
            'fake': None
        }
        tokendata = get_all_text_tokenized(row)

        dict_t['sentiment'] = sent.get_sentiment_by_sentences(tokendata['all']['sent'])
        dict_t['n_words'] = q.n_words(tokendata['all']['word'])
        dict_t['n_sentences'] = q.n_sentences(tokendata['all']['sent'])
        
        tagged_words = q.get_tags(tokendata['all']['word'])
        dict_t['pert_total_verbs'] = q.perct_verb_words(tagged_words)
        dict_t['pert_total_nouns'] = q.perct_noun_words(tagged_words)
        dict_t['pert_total_adj'] = q.perct_adj_words(tagged_words)
        dict_t['pert_total_conj_prep'] = q.perct_conj_words(tagged_words)

        sentiments = sent.get_pert_sentiment(tokendata['all']['word'])
        dict_t['pert_total_positive_words'] = sentiments[0]
        dict_t['pert_total_negative_words'] = sentiments[1]
        dict_t['pert_different_words'] = q.pert_diferent_words(tokendata['all']['word'])
        #dict_t['n_quotes'] = q.n_quotes(tokendata['all']['word'])
        dict_t['mean_words_per_sentence'] = q.mean_word_per_sent(tokendata['all']['word'])
        dict_t['mean_character_per_word'] = q.mean_characters_per_word(tokendata['all']['word'])


        dict_t['title_sentiment'] = sent.get_sentiment_by_sentences(tokendata['title']['sent'])
        dict_t['title_n_words'] = q.n_words(tokendata['title']['word'])
        dict_t['title_pert_total_conj_prep'] = q.perct_conj_words(tagged_words)

        sentiments = sent.get_pert_sentiment(tokendata['title']['word'])
        dict_t['title_pert_total_positive_words'] = sentiments[0]
        dict_t['title_pert_total_negative_words'] = sentiments[1]
        dict_t['fake'] = row['fake']
        style_dataset['articles'].append(dict_t)
    return style_dataset

# DATASET 3: Similarity
# Objective: Get info about diferences inside it content and the other documents
# Variables :
#   - Similarity between title and subtitle
#   - Similarity between title and text
#   - Similarity between text and subtitle
#   - TODO: Similarity between paragraphs in text (mean, min, max)
#   - TODO: Similarity between title of others
#   - TODO: Similarity between other texts
def get_similarity_dataset(dataset):
    similarity_dataset = {
        'articles': []
    }
    for _, row in dataset.iterrows():
        dict_t = {
            'sim_title_subtitle': None,
            'sim_title_text': None,
            'sim_text_subtitle': None,
            'sim_between_parapraphs': None,
            'fake': None
        }
        tokendata = get_all_text_tokenized(row)

        dict_t['sim_title_text'] = s.get_jaccard_similarity(tokendata['text']['joined_raw'], tokendata['title']['sent'][0])
        if row['subtitle'] == '':
            dict_t['sim_title_subtitle'] = None
            dict_t['sim_text_subtitle'] = None
        else:
            dict_t['sim_title_subtitle'] = s.get_jaccard_similarity(
                tokendata['title']['sent'][0], tokendata['subtitle']['sent'][0])
            dict_t['sim_text_subtitle'] = s.get_jaccard_similarity(
                 tokendata['text']['joined_raw'], tokendata['subtitle']['sent'][0])
        # Tokens
        #dict_t['sim_between_parapraphs'] = s.get_similarity_between_texts(tokendata['text']['paragraph'])
        dict_t['fake'] = row['fake']
        similarity_dataset['articles'].append(dict_t)
    return similarity_dataset
