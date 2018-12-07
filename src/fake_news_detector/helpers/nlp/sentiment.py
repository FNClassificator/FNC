from textblob.sentiments import NaiveBayesAnalyzer
from textblob.np_extractors import ConllExtractor
from textblob import TextBlob


def compute_sentiment(token_list, total): 
    positive_text = 0
    neutral_pol_text = 0
    negative_text = 0

    parcial_text  = 0
    neutral_subj_text = 0
    imparcial_text = 0
    for text in token_list:
        blob = TextBlob(text)

        pol = blob.sentiment.polarity
        if pol > 0.6: # Positive
            positive_text += 1
        elif pol < 0.4: # Negative
            negative_text += 1
        else: # Neutral
            neutral_pol_text += 1

        subj = blob.sentiment.subjectivity
        if subj > 0.6: # Parcial
            parcial_text += 1
        elif subj < 0.4: # Imparcial
            imparcial_text += 1
        else: # Neutral
            neutral_subj_text += 1

    dic = {
        'positive' : positive_text / total,
        'negative' : negative_text / total,
        'neutral_sent': neutral_pol_text / total,
        'parcial': parcial_text / total,
        'imparcial': imparcial_text / total,
        'neutral_subj' : neutral_subj_text / total
    }
    return dict

def get_sentiment_by_words(text):
    # Extract subjectivity and polarity
    blob = TextBlob(text)
    word_list = []
    total = len(blob.words)
    for word in blob.words:
        word_list.append(word)
    return compute_sentiment(word_list, total)
    
def get_sentiment_by_phrases(text):
    extractor = ConllExtractor()
    blob = TextBlob(text, np_extractor=extractor)

    text_list = blob.noun_phrases
    total = len(text_list)
    return compute_sentiment(text_list,total)

