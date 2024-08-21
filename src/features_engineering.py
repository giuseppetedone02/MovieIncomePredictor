import csv
import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from utils import plot_distribution


# Load the dataset
with open('../resources/dataset/Movie_dataset_cleaned.csv', mode='r', encoding='utf-8-sig') as movieCsv:
    reader = csv.DictReader(movieCsv)
    data = list(reader)

# Create a DataFrame from 'data'
df = pd.DataFrame(data)
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
for rating in ratings:
    df[rating] = (df['MPAA'] == rating).astype(int)


# Transformations on Runtime (using bins)
bins = [0, 90, 120, np.inf]
labels = ['<90', '90-120', '>120']
df['Runtime_Binned'] = pd.cut(df['Runtime'], bins=bins, labels=labels)

ordinal_mapping = {'<90': 0, '90-120': 1, '>120': 2}
df['Runtime_Encoded'] = df['Runtime_Binned'].map(ordinal_mapping)


# Transformations on Company, Director, Writer, Main Actor
# TO-DO: Aumentare numero di top per ogni categoria
# TO-DO: Cambiare campi in 'Numero di film' per ogni categoria
# top_companies = df['Company'].value_counts().nlargest(200).index
# top_directors = df['Director'].value_counts().nlargest(200).index
# top_writers = df['Writer'].value_counts().nlargest(200).index
# top_actors = df['Main Actor'].value_counts().nlargest(200).index

# df['Top_Company'] = df['Company'].isin(top_companies).astype(int)
# df['Top_Director'] = df['Director'].isin(top_directors).astype(int)
# df['Top_Writer'] = df['Writer'].isin(top_writers).astype(int)
# df['Top_Main_Actor'] = df['Main Actor'].isin(top_actors).astype(int)

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
for genre in genres:
    df[genre] = (df['Genre'] == genre).astype(int)


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
] + ratings + genres

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
