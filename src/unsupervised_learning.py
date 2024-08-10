import csv
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator

with open('../resources/dataset/Movie_dataset_features.csv', mode='r', encoding='utf-8-sig') as movieCsv:
    reader = csv.DictReader(movieCsv)
    dataset = list(reader)
    df = pd.DataFrame(dataset)

def elbow_method():
    # Create a list to store the inertia values
    inertia_values = []

    # Create a list of k values
    k_values = range(1, 11)

    # Iterate over the k values
    for k in k_values:
        # Create a KMeans model with k clusters
        model = KMeans(n_clusters=k, n_init=10, init='random')

        # Fit the model to the data
        model.fit(df)

        # Append the inertia value to the list
        inertia_values.append(model.inertia_)

    kneeLocator = KneeLocator(
        k_values, 
        inertia_values, 
        curve='convex', 
        direction='decreasing'
    )

    print(kneeLocator.elbow)

    # Plot the inertia values
    plt.plot(k_values, inertia_values)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()
    plt.savefig('../resources/plots/Elbow_Method.png')

    # Return the number of clusters
    return kneeLocator.elbow


def define_cluster():
    clusters = elbow_method()

    # Create a KMeans model with 3 clusters
    # then fits the model to the data
    model = KMeans(n_clusters=clusters, n_init=10, init='random')
    model.fit(df)

    # Add the cluster labels to the DataFrame and save it to a new CSV file
    df['Cluster'] = model.labels_
    df.to_csv('../resources/dataset/Movie_dataset_clusters.csv', index=False)

define_cluster()