from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize


def get_words_by_sentiment(text_token):
    analyzer = SentimentIntensityAnalyzer()
    positives = []
    negatives = []
    for word in text_token:
        vs = analyzer.polarity_scores(word)
        if vs['compound'] > 0.2:
            positives.append(word)
        elif vs['compound'] < -0.2:
            negatives.append(word)
    return positives, negatives

def get_pert_sentiment(text_token):
    sentiment = get_words_by_sentiment(text_token)
    positives = len(sentiment[0]) / len(text_token)
    negatives = len(sentiment[1]) / len(text_token)
    return positives, negatives


def get_sentiment_by_phrases(text):
    sentences = sent_tokenize(text)
    analyzer = SentimentIntensityAnalyzer()
    result = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        compound = analyzer.polarity_scores(sentence)['compound']
        result.append(compound)
    if result == []:
        return 0
    return sum(result)/len(result)

def get_sentiment_by_sentences(sentences):
    total = 0
    for sentence in sentences:
        total += get_sentiment_by_phrases(sentence)
    return total/len(sentences)

# For testing
if __name__ == "__main__":
    text = 'happy'
    get_sentiment_by_phrases(text)