
def get_common_nouns(tagged_text):
    total = []
    for word, tag in tagged_text:
        if tag.startswith('N'):
            if not tag in total:
                total.append(word)
    return total


def get_adj_words(tagged_text):
    total = []
    for word, tag in tagged_text:
        if tag.startswith('J'):
            if not tag in total:
                total.append(word)
    return total

def get_verb_words(tagged_text):
    total = []
    for word, tag in tagged_text:
        if tag.startswith('V'):
            if not tag in total:
                total.append(word)
    return total


def get_conj_words(tagged_text):
    total = []
    for word, tag in tagged_text:
        if tag.startswith('C'):
            if not tag in total:
                total.append(word)
    return total


#TODO
def get_noun_phrases_words(tagged_text):
    return