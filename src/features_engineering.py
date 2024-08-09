import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from utils import plot_distribution

# Read the dataset using 'with open'
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
top_companies = df['Company'].value_counts().nlargest(20).index
df['Top_Company'] = df['Company'].isin(top_companies).astype(int)

top_directors = df['Director'].value_counts().nlargest(20).index
df['Top_Director'] = df['Director'].isin(top_directors).astype(int)

top_writers = df['Writer'].value_counts().nlargest(20).index
df['Top_Writer'] = df['Writer'].isin(top_writers).astype(int)

top_actors = df['Main Actor'].value_counts().nlargest(20).index
df['Top_Main_Actor'] = df['Main Actor'].isin(top_actors).astype(int)

# Transformations on Budget (using log transformation)
# df['Log_Budget'] = np.log1p(df['Budget'])

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

# Convert True and False to 1 and 0
df = df.replace({True: 1, False: 0})

# Transformations on Score and Votes (using log transformation)
df['Log_Votes'] = np.log1p(df['Votes'])
min_max_scaler = MinMaxScaler()
df['Normalized_Score'] = min_max_scaler.fit_transform(df[['Score']])

print(df[['Score', 'Normalized_Score']].describe())
print(df[['Votes', 'Log_Votes']].describe())

# Selection of final features and creation of the final dataset (features + target)
features = [
    'Year', 'Decade', 'Recent', 'Runtime_Encoded', 'Top_Company', 'Top_Main_Actor', 
    'Top_Director', 'Top_Writer', 'Scaled_Standardized_Budget', 'Normalized_Score', 'Log_Votes'
] + ratings + genres

# Approximate the Worldwide Gross by removing the last two digits before the decimal point
df['Worldwide Gross'] = (df['Worldwide Gross'] // 100) * 100
df['Log_Worldwide_Gross'] = np.log1p(df['Worldwide Gross'])
print(df[['Worldwide Gross', 'Log_Worldwide_Gross']].describe())

final_df = df[features + ['Log_Worldwide_Gross']]

# Save the final dataset to a CSV file
final_df.to_csv('../resources/dataset/Movie_dataset_features.csv', index=False)


# Plot a graph showing the distribution of genres across various films
genre_distribution = df[genres].sum().sort_values()
plot_distribution(
    genre_distribution, 
    'Genre Distribution', 
    'Genre', 
    'Occurences', 
    '../resources/plots/distributions/genres_distribution.png'
)

# Plot a graph showing the distribution of ratings across various films
rating_distribution = df[ratings].sum().sort_values()
plot_distribution(
    rating_distribution, 
    'Rating Distribution', 
    'Rating', 
    'Occurences', 
    '../resources/plots/distributions/ratings_distribution.png'
)

# Plot a graph showing the distribution of runtime across various films
runtime_distribution = df['Runtime_Binned'].value_counts().sort_index()
plot_distribution(
    runtime_distribution, 
    'Runtime Distribution', 
    'Runtime', 
    'Occurences', 
    '../resources/plots/distributions/runtime_distribution.png',
    0
)
