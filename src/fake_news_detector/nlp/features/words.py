
import nltk

# T Y P E S

# N: Nouns
# C: Conjunctions
# J: Adjectives
# V: Verbs

def get_type_words(tagged_text,type):
    total = []
    for word, tag in tagged_text:
        if tag.startswith(type):
            total.append(word)
    return total

def get_unique_type_words(tagged_text, type):
    total = {}
    for word, tag in tagged_text:
        if tag.startswith(type):
            total[word] = True
    return list(total.keys())


# N O U N   P R H A S E S 

def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        yield subtree.leaves()


def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
    accepted = bool(2 <= len(word) <= 40
        and word.lower())
    return accepted

def get_noun_phrases(tagged_words):
    grammar = r"""
    NBAR:
        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        
    NP:
        {<NBAR>}
        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)
    tree = chunker.parse(tagged_words)
    result = []
    for leaf in leaves(tree):
        term = [ w for w,t in leaf if acceptable_word(w) ]
        phrase = ' '.join(term)
        result.append(phrase)
    return result