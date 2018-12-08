from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize


# def compute_sentiment(token_list, total): 
#     positive_text = 0
#     neutral_pol_text = 0
#     negative_text = 0

#     parcial_text  = 0
#     neutral_subj_text = 0
#     imparcial_text = 0
#     for text in token_list:
#         blob = TextBlob(text)

#         pol = blob.sentiment.polarity
#         if pol > 0.2: # Positive
#             positive_text += 1
#         elif pol < -0.2: # Negative
#             negative_text += 1
#         else: # Neutral
#             neutral_pol_text += 1

#         subj = blob.sentiment.subjectivity
#         if subj > 0.6: # Parcial
#             parcial_text += 1
#         elif subj < 0.4: # Imparcial
#             imparcial_text += 1
#         else: # Neutral
#             neutral_subj_text += 1

#     dict = {
#         'positive' : positive_text / total,
#         'negative' : negative_text / total,
#         'neutral_sent': neutral_pol_text / total,
#         'parcial': parcial_text / total,
#         'imparcial': imparcial_text / total,
#         'neutral_subj' : neutral_subj_text / total
#     }
#     return dict


def get_sentiment_by_phrases(text):

    sentences = sent_tokenize(text)

    analyzer = SentimentIntensityAnalyzer()
    result = []
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print(vs)
        # neg_scores.append(analyzer.polarity_scores(sentence)['neg'])
        # pos_scores.append(analyzer.polarity_scores(sentence)['pos'])
        compound = analyzer.polarity_scores(sentence)['compound']
        result.append(compound)
    return sum(result)/len(result)
    

if __name__ == "__main__":
    text = 'happy'
    get_sentiment_by_phrases(text)