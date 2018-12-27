from src.fake_news_detector.core.nlp import tokenize as tk

# TOKENIZE FOR DATASETS
def tokenize_colunm_of_text(dataset, label, stopwords):
    list_tokens = []
    for _, row in dataset.iterrows():
        tokens = clean_text_by_word(row[label], stopwords)
        list_tokens.append(tokens)
    return list_tokens

def tokenize_colunm_of_text_list(dataset, label, stopwords):
    list_tokens = []
    for _, row in dataset.iterrows():
        tokens = []
        for paragraph in row[label]:
            tokens += clean_text_by_word(paragraph, stopwords)
        list_tokens.append(tokens)
    return list_tokens


""" 
Do all process
1. Split by sentences
2. Split by word with important symbols
3. Delete symbol puntuations
4. Make lower case first word of sentence
5. Fold sentence
6. Lemmatize tokens
"""
def clean_text_by_sentence(text, stopwords):
    sentence_list = tk.tokenize_by_sentences(text)
    result = []
    for sent in sentence_list:
        token_list = tk.tokenize_by_treebank_word(sent)
        token_list = tk.remove_punctuations(token_list)
        token_list = tk.lemma_tokens(token_list)
        token_list = tk.to_lower(token_list)
        if stopwords:
            token_list = tk.remove_stopwords(token_list)
        # Join all in one
        clean_text = ' '.join(token_list)
        result.append(clean_text)
    return result


""" 
Do all process
1. tokenize by sentence
2. tokenize in words each sentence
"""
def clean_text_by_word(text, stopwords=True):
    result = []
    sentences = clean_text_by_sentence(text, stopwords)
    for sentence in sentences:
        words_list = tk.tokenize_by_treebank_word(sentence)
        result += words_list
    return result
