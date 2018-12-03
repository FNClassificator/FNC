from nltk.tokenize import sent_tokenize, word_tokenize, treebank
from nltk.corpus import stopwords, subjectivity
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import string


## 1. Tokenize by words
# Separte by all symbols
def tokenize_by_word(text):
    return word_tokenize(text)


# Separate by stricly words
def tokenize_by_treebank_word(text):
    tokenizer = treebank.TreebankWordTokenizer()
    return tokenizer.tokenize(text)


## 2. Tokenize by sentence
def tokenize_by_sentences(text):
    return sent_tokenize(text)


def get_class_text(text):
    return nltk.Text(text)


# Utils functions with tokenized objects-------------------------------------

# Clean punctuation symbols
# Return the same tokens without all punctuation symbols
def remove_punctuations(tokens):
    words = [word for word in tokens if word.isalpha()]
    return words

# Clean stop words
# Return the same tokens without the list of tokens considered stop words by nltk
def remove_stopwords(words):
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    return words


#  Steam tokens
def steam_words(tokens):
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    return stemmed


# Lemmatize tokens
def lemma_tokens(tokens):
    w_lemmatizer = WordNetLemmatizer()
    lemmatized = [w_lemmatizer.lemmatize(word, pos='v') for word in tokens]
    return lemmatized


# Get tags
def get_tags(tokens):
    tags = [pos_tag(word) for word in tokens]
    return tags


# Get words in same context
# Class is nltk.Text
def findall(text, pre, post):
    list_text = text.findall("<"+pre+">(<.*>)<"+post+">")
    return list_text


# Returns all sentences with same word
# Pre: text in nltk.Text
def find_context(text, word):
    return text.concordance(word)


# Palabras del mismo contexto
def get_similars(text, word):
    return text.similar(word)
