import spacy

# DETECT SUBJECT

def get_subject(text):
    nlp = spacy.load('en')
    sent = "I shot an elephant"
    doc=nlp(sent)
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj") ]
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