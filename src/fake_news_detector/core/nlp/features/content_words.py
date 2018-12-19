
def pert_modal_verbs(text_tokens):
    modals = ['can', 'could', 'may', 'might', 'must', 'will', 'should', 'would']
    suma = 0
    for text in text_tokens:
        if text in modals:
            suma += 1
    return suma/len(text_tokens)


# TODO: Difficult, it has to have a model training
def pert_certainty_terms(text_tokens):
    # DICCIONARY
    return

# TODO: Difficult, it has to have a model training
def pert_generalizing_terms(text_tokens):
    # Find subject and classify
    return

# TODO: Difficult, it has to have a model training
def pert_tentative_terms(text_tokens):
    # DICCIONARY
    return

# TODO: Medium
def pert_numbers_and_quantifiers(text_tokens):
    return

# TODO: Easy
def pert_question_marks(text_tokens):
    return