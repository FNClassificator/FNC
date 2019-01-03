
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# DOC2VEC FUNCTIONS
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