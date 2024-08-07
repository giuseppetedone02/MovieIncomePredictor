import csv
import pandas as pd
import numpy as np

# Read the dataset using 'with open'
with open('./resources/Movie_dataset_cleaned.csv', mode='r', encoding='utf-8-sig') as movieCsv:
    reader = csv.DictReader(movieCsv)
    data = list(reader)

# Create a DataFrame from 'data'
df = pd.DataFrame(data)

# Convert numeric columns that might be read as strings
numeric_columns = ['Year', 'Runtime', 'Budget', 'Worldwide Gross', 'Score', 'Votes']
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Transformations on Year (using decade and recent flag)
df['Decade'] = (df['Year'] // 10) * 10
df['Recent'] = df['Year'] >= 2010

# Transformations on MPAA (categorical feature)
ratings = ['G', 'PG', 'PG-13', 'R', 'NC-17', 'Not Rated', 'Unrated', 'X']
for rating in ratings:
    df[rating] = (df['MPAA'] == rating).astype(int)

# Transformations on Runtime (using bins)
bins = [0, 60, 90, 120, np.inf]
labels = ['<60', '60-90', '90-120', '>120']
df['Runtime_Binned'] = pd.cut(df['Runtime'], bins=bins, labels=labels)

# Transformations on Company, Director, Writer, Main Actor
top_companies = df['Company'].value_counts().nlargest(20).index
df['Top_Company'] = df['Company'].isin(top_companies)

top_directors = df['Director'].value_counts().nlargest(20).index
df['Top_Director'] = df['Director'].isin(top_directors)

top_writers = df['Writer'].value_counts().nlargest(20).index
df['Top_Writer'] = df['Writer'].isin(top_writers)

top_actors = df['Main Actor'].value_counts().nlargest(20).index
df['Top_Main_Actor'] = df['Main Actor'].isin(top_actors)

# Transformations on Budget (using log transformation)
df['Log_Budget'] = np.log1p(df['Budget'])

# Transformations on Genre (categorical feature)
genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama', 'Fantasy', 'Historical', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
for genre in genres:
    df[genre] = (df['Genre'] == genre).astype(int)

# Transformations on Score and Votes (using log transformation)
df['Log_Votes'] = np.log1p(df['Votes'])
df['Weighted_Score'] = df['Score'] * df['Votes']

# Selection of final features
features = [
    'Year', 'Decade', 'Recent', 'Runtime_Binned', 'Top_Company', 'Top_Main_Actor', 
    'Top_Director', 'Top_Writer', 'Log_Budget', 'Weighted_Score', 'Log_Votes'
] + ratings + genres

# Creation of the final dataset for the model (features + target)
final_df = df[features + ['Worldwide Gross']]
print(final_df.head())

# Save the final dataset to a CSV file
final_df.to_csv('./resources/Movie_dataset_features.csv', index=False)