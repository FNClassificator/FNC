
import nltk

def get_tags(token_text):
    return nltk.pos_tag(token_text)

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
    # Call Stanford tree
    # Split tree by clauses by preposition
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


# U N C E R T A I N T Y

def pert_modal_verbs(text_tokens):
    modals = ['can', 'could', 'may', 'might', 'must', 'will', 'should', 'would']
    suma = 0
    for text in text_tokens:
        if text in modals:
            suma += 1
    return suma/len(text_tokens)


def pert_certainty_terms(text_tokens):
    # DICCIONARY
    return


def pert_generalizing_terms(text_tokens):
    # DICCIONARY
    return


def pert_tentative_terms(text_tokens):
    # DICCIONARY
    return


def pert_numbers_and_quantifiers(text_tokens):
    return


def pert_question_marks(text_tokens):
    return

# T Y P E   O F   V E R B S

def pert_subjective_verbs(texts_tokens):
    total_verbs = 0
    total_subj_verbs = 0
    return


def pert_report_verbs(text_tokens):
    return


def pert_factive_verbs(text_tokens):
    return


def pert_imperative_commands(text_tokens):
    return

# S E N T I M E N T

def pert_positive_words(text_tokens):
    return

def pert_negative_words(text_tokens):
    return

def pert_emotional_phrases(text_tokens):
    return


# D I V E R S I T Y

def lexical_diversity(text_tokens):
    return


def reduncancy(text_tokens):
    return


# NON - I M M E D I A C Y

def pert_passive_voice(text_tokens):
    return

def pert_retorical_questions(text_tokens):
    return

def pert_self_reference(text_tokens):
    return

def pert_group_reference(text_tokens):
    return

def perct_other_reference(text_tokens):
    return