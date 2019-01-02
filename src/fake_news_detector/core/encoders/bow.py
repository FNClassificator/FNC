
from gensim.corpora import Dictionary

# MAIN FUNCTION ..............
def bow_encoding(data, no_above, no_below):
    dictionary = create_dictionary(data)
    filter_by_freq(dictionary,no_above=no_above, no_below=no_below)
    corpus = data2bow(dictionary,data)
    return corpus, dictionary
# ............................


# BOW FUNCTIONS
def create_dictionary(data):
    return Dictionary(documents=data)

def filter_by_freq(dictionary, no_above=0.8, no_below=3):
    dictionary.filter_extremes(no_above=no_above, no_below=no_below)
    dictionary.compactify()
    return

def data2bow(dictionary, data):
    bow_encoding = []
    for document in data:
        bow_encoding.append(dictionary.doc2bow(document))
    return bow_encoding

# DICTIONARY INFO

def print_size(dictionary):
    print("Found {} words.".format(len(dictionary.values())))