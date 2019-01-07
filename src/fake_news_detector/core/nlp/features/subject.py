import spacy
import nltk
from  src.fake_news_detector.core.nlp import chunking as c
from  src.fake_news_detector.core.nlp.features import quantity as q
# DETECT SUBJECT


def get_subject(text):
    tree = c.standford_parse_tree(text)

    subject = None
    for elem in tree[0]:
        if elem.label() == 'NP':
            subject = elem

    if subject == None:
        return []
    result = []
    for word in subject.leaves():
        result.append(word)
    return result

def get_subjects(sentences):
    result = []
    for sent in sentences:
        result += get_subject(sent)
    return result

def detect_subject(text):
    # No tiene sujecto
    subject = get_subject(text)
    if subject == []:
        return True
    # 3. Usa It
    if 'it' in subject:
        return True
    # 5. Usa ONE or You
    if 'one' in subject:
        return True
    if 'you' in subject:
        return True
    if pert_self_reference(subject):
        return True
    if q.perct_verb_words(subject) > 0 and len(subject) < 3:
        return True
    if len(subject) < 2:
        return True
    return False


# OTHER SUBJECT PROPERTIES
def pert_self_reference(subject):
    list = ['this', 'I']
    for item in list:
        if item in subject:
            return True
    return False
