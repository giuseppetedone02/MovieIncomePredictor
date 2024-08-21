import csv
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import smogn
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator

from oversampling import smogn_resample_data


# Function to determine the optimal number of clusters using the elbow method
def elbow_method(y):
    # Create a list to store the inertia values
    inertia_values = []
    k_values = range(1, 11)

    # Iterate over the k values
    for k in k_values:
        # Create a KMeans model with k clusters
        model = KMeans(n_clusters=k, n_init=10, init='random')

        # Fit the model to the Log_Worldwide_Gross column
        model.fit(y)

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


def define_cluster(df, features, titlePrefix='', suffix=''):
    y = df[features]
    clusters = elbow_method(y)

    # Create a KMeans model with the optimal number of clusters
    # then fits the model to the Log_Worldwide_Gross column
    model = KMeans(n_clusters=clusters, n_init=10, init='random')
    model.fit(y)

    # Add the cluster labels to the DataFrame and save it to a new CSV file
    clusters = model.labels_
    centroids = model.cluster_centers_

    df['Cluster'] = model.labels_
    df.to_csv(f'../resources/dataset/Movie_dataset_clusters.csv', index=False)
    
    visualize_pie_chart(df, titlePrefix, suffix)

    return clusters, centroids


# Display a pie chart of the distribution of movies in clusters
def visualize_pie_chart(dataFrame, features, titlePrefix='', suffix=''):
    cluster_counts = dataFrame['Cluster'].value_counts()

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        cluster_counts, 
        labels=cluster_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
    )
    plt.title(f'{titlePrefix} Distribution of Movies in Clusters')
    plt.legend(
        wedges, 
        [f'Cluster {i}' for i in cluster_counts.index], 
        title="Clusters", 
        loc="lower left" 
    )
    plt.savefig(f'../resources/plots/unsupervised_learning/clusters_pie_chart{suffix}.png')
    plt.show()



# Load the dataset
df = pd.read_csv('../resources/dataset/Movie_dataset_features.csv', encoding='utf-8-sig')

features = [
    'Runtime_Encoded', 'Director_Num_Movies', 'Writer_Num_Movies', 'Main_Actor_Num_Movies', 'Scaled_Standardized_Budget', 
    'Decade', 'G', 'PG', 'PG-13', 'R', 'NC-17', 'Not Rated', 'Unrated', 'Action', 'Adventure', 'Animation', 'Comedy', 
    'Crime', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'Western'
]
clusters, centroids = define_cluster(df, features)

# Calculate and print the silhouette score
sil_score = silhouette_score(df[features], clusters)
print(f'Silhouette Score: {sil_score}')

# Calculate mean and median of features for each cluster
cluster_stats = df.groupby('Cluster')[features].agg(['mean', 'median'])
print(cluster_stats)