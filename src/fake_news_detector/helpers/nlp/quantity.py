


""" ABOUT WORDS """

def n_words(text_tokens):
    return len(text_tokens)

def perct_noun_words(tag_tokens):
    total = 0
    for token, tag in tag_tokens:
        if tag == 'NN':
            total += 1
    return total / len(tag_tokens)


def perct_adj_words(tag_tokens):
    total = 0
    for _, tag in tag_tokens:
        if tag.startswith('J'):
            total += 1
    return total / len(tag_tokens)


def perct_verb_words(tag_tokens):
    total = 0
    for _, tag in tag_tokens:
        if tag.startswith('V'):
            total += 1
    return total / len(tag_tokens)

""" ABOUT SENTENCES """

def n_sentences(sent_tokens):
    return len(sent_tokens)

# A V E R A G E S

def mean_word_per_sent(sent_tokens):
    t_sum = 0
    for sent in sent_tokens:
        t_sum += len(sent)
    return t_sum / len(sent_tokens)

def mean_clause_per_sent(sent_tokens):
    t_sum = 0
    # call function to split by clauses
    return t_sum

def mean_characters_per_word(texts_tokens):
    t_sum = 0
    total = 0
    for sent in texts_tokens:
        for word in sent:
            total += 1
            t_sum += len(word)
    return t_sum / total

def mean_punctuation_per_sent(sent_tokens):
    # TODO
    return 0

# T Y P E   O F   V E R B S

def pert_subjective_verbs(texts_tokens):
    total_verbs = 0
    total_subj_verbs = 0
    for sent in texts_tokens:
        