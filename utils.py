import matplotlib.pyplot as plt

def plot_distribution(data, title, xlabel, ylabel, filename):
    plt.figure(figsize=(7, 10))
    data.plot(kind='bar')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    plt.savefig(filename)