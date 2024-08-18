import csv
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator


# Load the dataset
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

        # Fit the model to the Log_Worldwide_Gross column
        model.fit(df[['Log_Worldwide_Gross']])

        # Append the inertia value to the list
        inertia_values.append(model.inertia_)

    # Create a KneeLocator object to find the optimal k value
    kneeLocator = KneeLocator(
        k_values, 
        inertia_values, 
        curve='convex', 
        direction='decreasing'
    )

    # Plot the inertia values
    plt.plot(k_values, inertia_values)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.savefig('../resources/plots/unsupervised_learning/elbow_method.png')
    plt.show()

    # Return the number of clusters
    return kneeLocator.elbow


def define_cluster():
    clusters = elbow_method()

    # Create a KMeans model with the optimal number of clusters
    # then fits the model to the Log_Worldwide_Gross column
    model = KMeans(n_clusters=clusters, n_init=10, init='random')
    model.fit(df[['Log_Worldwide_Gross']])

    # Add the cluster labels to the DataFrame and save it to a new CSV file
    df['Cluster'] = model.labels_
    df.to_csv('../resources/dataset/Movie_dataset_clusters.csv', index=False)
    visualize_pie_chart(df)


# Display a pie chart of the distribution of movies in clusters
def visualize_pie_chart(dataFrame):
    cluster_counts = dataFrame['Cluster'].value_counts()

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        cluster_counts, 
        labels=cluster_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
    )
    plt.title('Distribution of Movies in Clusters (Based on Worldwide Gross)')
    plt.legend(
        wedges, 
        [f'Cluster {i}' for i in cluster_counts.index], 
        title="Clusters", 
        loc="lower left" 
    )
    plt.savefig('../resources/plots/unsupervised_learning/clusters_pie_chart.png')
    plt.show()

define_cluster()