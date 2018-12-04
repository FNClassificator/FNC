import nltk
from textblob import TextBlob

# Variables

def count_common_nouns(token_text):
    text_tagged = nltk.pos_tag(token_text)
    total = 0
    for _, tag in text_tagged:
        if tag == 'NN':
            total += 1
    return total/len(token_text)

def count_adjectives(token_text):
    text_tagged = nltk.pos_tag(token_text)
    total = 0
    for _, tag in text_tagged:
        if tag.startswith('J'):
            total += 1
    return total/len(token_text)

def get_sentiment(text):
    return TextBlob(text).sentiment


def get_lenght(token_text):
    return len(token_text)

# Subclassificators