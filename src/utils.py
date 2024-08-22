import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_distribution(data, title, xlabel, ylabel, filename, rotation=45):
    plt.figure(figsize=(7, 10))
    data.plot(kind='bar')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)

    plt.savefig(filename)

def check_type(train, test):
    if not np.issubdtype(train.dtype, np.number):
        train = pd.to_numeric(train, errors='coerce')
    if not np.issubdtype(test.dtype, np.number):
        test = pd.to_numeric(test, errors='coerce')

def reduce_categories(df, column, threshold=10):
    counts = df[column].value_counts()
    to_keep = counts[counts > threshold].index
    df[column] = df[column].apply(lambda x: x if x in to_keep else 'Other')
