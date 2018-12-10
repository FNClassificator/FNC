from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

# TODO
def get_positive_words(text):
    return


# TODO
def get_negative_words(text):
    return


def pert_positive_words(text):
    return

def pert_negative_words(text):
    return


def get_sentiment_by_phrases(text):

    sentences = sent_tokenize(text)

    analyzer = SentimentIntensityAnalyzer()
    result = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        compound = analyzer.polarity_scores(sentence)['compound']
        result.append(compound)
    return sum(result)/len(result)
    
# For testing
if __name__ == "__main__":
    text = 'happy'
    get_sentiment_by_phrases(text)