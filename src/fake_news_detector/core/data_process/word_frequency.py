from nltk import FreqDist
import numpy as np
import matplotlib.pyplot as plt
from src.fake_news_detector.core.nlp.features import similarity as sim

def get_all_words(word_list):
    all_words = []
    for item in word_list:
        for word in item:
            all_words.append(word)
    return all_words

def get_word_freq(word_list, output, topn=300):
    # 1. Get list of all words
    all_words = get_all_words(word_list)
    # 2. Use nltk word distribuition
    fdist = FreqDist(all_words)
    
    # 3. Get Unique words
    unique_w = len(fdist)
    #4. Top n Words
    topn_w = fdist.most_common(topn)
    
    if output:
        print('Number of unique words: ', unique_w)
        print('Top', topn, 'words:' )
        print(topn_w)
    return fdist

def print_frequency_distribution( fdist, topn):
    print('Number of unique words: ', len(fdist))
    print('Top', topn, 'words:' )
    top_topn_words = fdist.most_common(topn)
    print(top_topn_words)


def pert_of_difference(fdist_one, fdist_two, topn=300):
    top_words_one = fdist_one.most_common(topn)
    top_words_two = fdist_two.most_common(topn)
    top_words_one_f = [w[0] for w in top_words_one]
    top_words_two_f = [w[0] for w in top_words_two]
    similarity = sim.get_jaccard_similarity(top_words_one_f, top_words_two_f)
    return similarity


def doc_length_distribution(dataset):
    # Distribution of doc length
    doc_lengths = list(dataset['doc_length'])
    num_bins = 1000
    fig, ax = plt.subplots(figsize=(12,6))
    # the histogram of the data
    n, bins, patches = ax.hist(doc_lengths, num_bins, normed=1)
    ax.set_xlabel('Document Length (tokens)', fontsize=15)
    ax.set_ylabel('Normed Frequency', fontsize=15)
    ax.grid()
    ax.set_xticks(np.logspace(start=np.log10(50),stop=np.log10(2000),num=8, base=10.0))
    plt.xlim(0,2000)
    ax.plot([np.average(doc_lengths) for i in np.linspace(0.0,0.0035,100)], np.linspace(0.0,0.0035,100), '-',
            label='average doc length')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.show()



