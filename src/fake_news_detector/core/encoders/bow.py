from gensim.corpora import Dictionary


##############################
######## BOW ENCODING ########
##############################

# MAIN FUNCTION

""" 
Vectorize Text to BOW
@corpus: Corpus is all the text to encode
@filter_by_freq: If true, dictionary only includes most frequent words
return: all documents encoded by BOW and the dictionary
"""
def run_BOW(corpus, filter_by_freq, output):
    #1. Create Dictionary
    dictionary = v.text2Dictionary(corpus)
    if output:
        v.dictionary_info(dictionary)
    #2. Filter by freq
    if filter_by_freq:
        v.filter_most_freq_words(dictionary)
        if output:
            v.dictionary_info(dictionary)
    
    bow_encoding = doc2BOW(corpus, dictionary)
    return bow_encoding, dictionary

# PROCESS FUNCTIONS

""" 
Vectorize Text to Dictionary
@corpus: Corpus is all the text to encode
"""
def text2Dictionary(corpus):
    return Dictionary(documents=corpus)

def doc2BOW(corpus, dictionary):
    encoded_corpus = []
    for elem in corpus: # Elem has to be a list of tokens
        encoded_corpus.append(dictionary.doc2bow(elem))
    return encoded_corpus

def filter_most_freq_words(dictionary, noabove=0.8, nobelow=3):
    dictionary.filter_extremes(no_above=noabove, no_below=nobelow)
    dictionary.compactify()  # Reindexes the remaining words after filtering
    

# INFO FUNCTIONS

def dictionary_info(dictionary):
    print("Found {} words.".format(len(dictionary.values())))