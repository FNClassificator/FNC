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

def correlation(X, y):
    cmap = cm.get_cmap('gnuplot')
    scatter = scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
    return scatter