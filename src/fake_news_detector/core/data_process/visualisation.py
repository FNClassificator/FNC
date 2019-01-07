import seaborn as sns
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm

def countplot(data_column, label_col):
    sns.countplot(data_column,label=label_col)
    plt.show()

def boxplot(dataset, label):
    dataset.drop(label, axis=1, title_plot)
            .plot(kind='box', 
                    subplots=True, 
                    sharex=False, 
                    sharey=False, 
                    figsize=(9,9), 
                    title=title_plot)
    plt.savefig('dataset_box')
    plt.show()

def histogram(dataset, label, title):
    dataset.drop(label, axis=1).hist(bins=30, figsize=(9,9))
    plt.show()

def double_barplot(data, label_one, label_two, title):
    fig, ax = plt.subplots(1,1,figsize=(20,5))

    nr_top_words = len(data)
    nrs = list(range(nr_top_words))

    sns.barplot(nrs, data[label_one].values, color='b', ax=ax, label=label_one)
    sns.barplot(nrs, data[label_two], color='g', ax=ax, label=label_two)

    ax.set_title(title, fontsize=16)
    ax.legend(prop={'size': 16})
    ax.set_xticks(nrs)
    ax.set_xticklabels(data.index, fontsize=14, rotation=90)

def displot(data, label_x, title):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(label_x)
    sns.distplot(data, bins=50, ax=ax)

def correlation(X, y):
    cmap = cm.get_cmap('gnuplot')
    scatter = scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
    return scatter