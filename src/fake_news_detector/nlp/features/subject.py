import spacy

from  src.fake_news_detector.nlp import chunking as c
# DETECT SUBJECT

def get_subject(text):
    tree = c.standford_parse_tree(text)
    return sub_toks

def detect_subject(text):
    sub_tokens = get_subject(text)
    # 1. No tiene sujeto
    if sub_tokens == []:
        return True

    # 3. Usa It
    if 'it' in sub_tokens:
        return True
    # 4. Passive voice (no es impersonal pero suele generalizar)

    # 5. Usa ONE or You
    if 'one' in sub_tokens:
        return True
    if 'you' in sub_tokens:
        return True
    return


# OTHER SUBJECT PROPERTIES

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