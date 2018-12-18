from src.fake_news_detector.core.nlp import tokenize as tk


""" 
Do all process
1. Split by sentences
2. Split by word with important symbols
3. Delete symbol puntuations
4. Make lower case first word of sentence
5. Fold sentence
6. Lemmatize tokens
"""
def clean_text_by_sentence(text):
    sentence_list = tk.tokenize_by_sentences(text)
    result = []
    for sent in sentence_list:
        token_list = tk.tokenize_by_treebank_word(sent)
        token_list = tk.remove_punctuations(token_list)
        token_list = tk.lemma_tokens(token_list)

        # Lower case first word
        if token_list:
            word = list(token_list[0])
            word[0] = word[0].lower()   
            word = ''.join(word)
            token_list[0] = word
            # Join all in one
            clean_text = ' '.join(token_list)
            result.append(clean_text)
    return result


""" 
Do all process
1. tokenize by sentence
2. tokenize in words each sentence
"""
def clean_text_by_word(text):
    result = []
    sentences = clean_text_by_sentence(text)
    for sentence in sentences:
        words_list = tk.tokenize_by_treebank_word(sentence)
        result += words_list
    return result
