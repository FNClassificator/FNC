
import nltk


# Input: List of words
def get_tags(token_text): # Aux
    return nltk.pos_tag(token_text)

# C O U N T

def n_words(text_tokens):
    return len(text_tokens)

def n_sentences(sent_tokens):
    return len(sent_tokens)

# Input: List of tagged words
def perct_noun_words(tag_tokens):
    total = 0
    for _, tag in tag_tokens:
        if tag.startswith('N'):
            total += 1
    return total / len(tag_tokens)

# Input: List of tagged words
def perct_adj_words(tag_tokens):
    total = 0
    for _, tag in tag_tokens:
        if tag.startswith('J'):
            total += 1
    return total / len(tag_tokens)

# Input: List of tagged words
def perct_verb_words(tag_tokens):
    total = 0
    for _, tag in tag_tokens:
        if tag.startswith('V'):
            total += 1
    return total / len(tag_tokens)

# Input: List of tagged words
def perct_conj_words(tag_tokens):
    total = 0
    for _, tag in tag_tokens:
        if tag.startswith('C'):
            total += 1
    return total / len(tag_tokens)


# C O M P L E X I T Y

def mean_word_per_sent(sent_tokens):
    t_sum = 0
    for sent in sent_tokens:
        t_sum += len(sent)
    return t_sum / len(sent_tokens)


def mean_characters_per_word(texts_tokens):
    t_sum = 0
    total = 0
    for sent in texts_tokens:
        for word in sent:
            total += 1
            t_sum += len(word)
    return t_sum / total


def mean_clause_per_sent(sent_tokens):
    t_sum = 0
    # Call Stanford tree
    # Get length of first level of tree from root
    return t_sum


def freq_distribution(text_token):
    fdist1 = nltk.FreqDist(text_token)
    filtered_word_freq = dict((word, freq) for word, freq in fdist1.items() if not word.isdigit())
    return filtered_word_freq


def lexical_diversity(word_freq):
    return len(word_freq)


def pert_diferent_words(text_token):
    result = []
    for text in text_token:
        if text in result:
            pass
        else:
            result.append(text)
    return len(result)/len(text_token)






