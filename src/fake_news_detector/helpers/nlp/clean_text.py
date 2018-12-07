from src.fake_news_detector.helpers.nlp import tokenize as tk


""" 
Do all process
1. Split by sentences
2. Split by word with important symbols
3. TODO: Delete symbol puntuations
4. Lemmatize tokens
"""
def clean_text_by_sentence(text):
    sentence_list = tk.tokenize_by_sentences(text)
    result = []
    for sent in sentence_list:
        token_list = tk.tokenize_by_treebank_word(sent)
        token_list = tk.remove_punctuations(token_list)
        token_list = tk.remove_stopwords(token_list)
        token_list = tk.lemma_tokens(token_list)
        result.append(token_list)
    return result


""" 
Do all process
1. Split by words with important symbols
2. TODO: Delete symbol puntuations
3. Lemmatize tokens
"""
def clean_text_words(text):
    words_list = tk.tokenize_by_treebank_word(text)
    result = tk.remove_punctuations(words_list)
    result = tk.remove_stopwords(result)
    result = tk.lemma_tokens(result)
    return result
