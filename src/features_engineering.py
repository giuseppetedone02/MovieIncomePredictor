import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# Plot the distribution of a given feature
def plot_distribution(data, title, xlabel, ylabel, filename, rotation=45):
    plt.figure(figsize=(7, 10))
    data.plot(kind='bar')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)

    plt.savefig(filename)


# Load the dataset
df = pd.read_csv('../resources/dataset/Movie_dataset_cleaned.csv', encoding='utf-8-sig')
df = df.dropna()

# Convert numeric columns that might be read as strings
numeric_columns = [
    'Year', 'Runtime', 
    'Budget', 'Worldwide Gross', 
    'Score', 'Votes'
]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')


# Transformations on Year (using decade and recent flag)
df['Decade'] = (df['Year'] // 10) * 10
df['Recent'] = df['Year'] >= 2010


# Transformations on MPAA (categorical feature)
ratings = [
    'G', 'PG', 'PG-13', 
    'R', 'NC-17', 'Not Rated', 
    'Unrated', 'X'
]
df = pd.get_dummies(df, columns=['MPAA'], prefix='', prefix_sep='')


# Transformations on Runtime (using bins)
bins = [0, 90, 120, np.inf]
labels = ['<90', '90-120', '>120']
df['Runtime_Binned'] = pd.cut(df['Runtime'], bins=bins, labels=labels)

ordinal_mapping = {'<90': 0, '90-120': 1, '>120': 2}
df['Runtime_Encoded'] = df['Runtime_Binned'].map(ordinal_mapping)


# Transformations on Company, Director, Writer, Main Actor
# Compute the number of movies for each category
df['Director_Num_Movies'] = df['Director'].map(df['Director'].value_counts())
df['Writer_Num_Movies'] = df['Writer'].map(df['Writer'].value_counts())
df['Main_Actor_Num_Movies'] = df['Main Actor'].map(df['Main Actor'].value_counts())
df['Company_Num_Movies'] = df['Company'].map(df['Company'].value_counts())


# Transformations on Budget (using standardization)
scaler = StandardScaler()
df['Standardized_Budget'] = scaler.fit_transform(df[['Budget']])

# Apply Min-Max Scaling to Standardized_Budget to avoid negative values
min_max_scaler = MinMaxScaler()
df['Scaled_Standardized_Budget'] = min_max_scaler.fit_transform(df[['Standardized_Budget']])


# Transformations on Genre (categorical feature)
genres = [
    'Action', 'Adventure', 'Animation', 'Comedy', 
    'Crime', 'Drama', 'Fantasy', 
    'Horror', 'Mystery', 'Romance', 
    'Sci-Fi', 'Thriller', 'Western'
]
df = pd.get_dummies(df, columns=['Genre'], prefix='', prefix_sep='')


# Transformations on Country (categorical feature)
df = pd.get_dummies(df, columns=['Country'], prefix='Country', prefix_sep='_')


# Transformations on True and False values
# Imposta l'opzione per evitare il downcasting silenzioso
pd.set_option('future.no_silent_downcasting', True)

# Convert True and False to 1 and 0
df = df.replace({True: 1, False: 0}).infer_objects(copy=False)


# Transformations on Score and Votes (using log transformation)
df['Log_Votes'] = np.log1p(df['Votes'])
min_max_scaler = MinMaxScaler()
df['Normalized_Score'] = min_max_scaler.fit_transform(df[['Score']])

# Selection of final features and creation of the final dataset (features + target)
features = [
    'Year', 'Decade', 'Recent', 'Runtime_Encoded', 'Director_Num_Movies', 'Writer_Num_Movies', 
    'Main_Actor_Num_Movies', 'Company_Num_Movies', 'Scaled_Standardized_Budget', 
    'Normalized_Score', 'Log_Votes'
] + ratings + genres + list(df.filter(like='Country_').columns)

# Approximate the Worldwide Gross by removing the last two digits before the decimal point
df['Worldwide Gross'] = (df['Worldwide Gross'] // 100) * 100
df['Log_Worldwide_Gross'] = np.log1p(df['Worldwide Gross'])

final_df = df[features + ['Log_Worldwide_Gross']]

# Save the final dataset to a CSV file
final_df.to_csv('../resources/dataset/Movie_dataset_features.csv', index=False)


# Plot a graph showing the distribution of 
# genres, ratings and runtime across various films
genre_distribution = df[genres].sum().sort_values()
plot_distribution(
    genre_distribution, 
    'Genre Distribution', 
    'Genre', 
    'Occurences', 
    '../resources/plots/distributions/genres_distribution.png'
)

rating_distribution = df[ratings].sum().sort_values()
plot_distribution(
    rating_distribution, 
    'Rating Distribution', 
    'Rating', 
    'Occurences', 
    '../resources/plots/distributions/ratings_distribution.png'
)

runtime_distribution = df['Runtime_Binned'].value_counts().sort_index()
plot_distribution(
    runtime_distribution, 
    'Runtime Distribution', 
    'Runtime', 
    'Occurences', 
    '../resources/plots/distributions/runtime_distribution.png',
    0
)
