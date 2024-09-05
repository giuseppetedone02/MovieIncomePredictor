import csv
import pandas as pd
from pyswip import Prolog


# Function to remove rows with missing values from a dataset
def remove_na_from_dataset(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path, on_bad_lines='skip')
    df_cleaned = df.dropna()
    df_cleaned.to_csv(output_csv_path, index=False)

# Function to write facts about genres and ratings to kb.pl from the dataset
def write_genres_and_ratings(dataset_path):
    df = pd.read_csv(dataset_path)
    
    genres = df['Genre'].unique()
    ratings = df['MPAA'].unique()

    with open('kb.pl', mode='w', encoding='utf-8') as kbFile:
        for genre in genres:
            kbFile.write(f"genre('{genre}').\n")
        
        kbFile.write('\n')

        for rating in ratings:
            kbFile.write(f"rating('{rating}').\n")

        kbFile.write('\n')

# Function to write facts about movies to kb.pl from the dataset
def write_movie_info(dataset_path):
    with open(dataset_path, mode='r', encoding='utf-8-sig') as movieCsv:
        reader = csv.DictReader(movieCsv)

        with open('kb.pl', mode='a', encoding='utf-8') as kbFile:
            for row in reader:
                title = repr(row['Title'])
                year = row['Year']
                mpaa = repr(row['MPAA'])
                runtime = row['Runtime']
                company = repr(row['Company'])
                country = repr(row['Country'])
                director = repr(row['Director'])
                writer = repr(row['Writer'])
                main_actor = repr(row['Main Actor'])
                budget = row['Budget'].replace(',', '')
                worldwide_gross = row['Worldwide Gross'].replace(',', '')
                genre = repr(row['Genre'])
                score = row['Score']
                votes = row['Votes']

                kbFile.write(f"movie({title}, {year}, {mpaa}, {runtime}, {company}, {country}, "
                             f"{director}, {writer}, {main_actor}, {budget}, {worldwide_gross}, "
                             f"{genre}, {score}, {votes}).\n")

# Function to write rules to kb.pl
def write_rules():
    rules = """
    roi(Movie, ROI) :-
        movie(Movie, _, _, _, _, _, _, _, _, Budget, Gross, _, _, _),
        ROI is ((Gross - Budget) / Budget) * 100.

    success(Movie) :-
        roi(Movie, ROI),
        ROI > 300.
    """

    with open('kb.pl', mode='a', encoding='utf-8') as kbFile:
        for rule in rules.split('.'):
            if rule.strip() != '':
                kbFile.write(rule + '.\n')

# Function to write the entire knowledge base
def write_knowledge_base(dataset_path):
    write_genres_and_ratings(dataset_path)
    write_movie_info(dataset_path)
    write_rules()

# Function to execute queries in Prolog and return the results
def execute_query(query):
    result = list(prolog.query(query))
    return result


# Function to update the original dataset with new columns based on Prolog rules
def update_dataset_with_prolog_results(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    success_col = []
    roi_col = []

    for title in df['Title']:
        title_escaped = repr(title)
        success_query = f"success({title_escaped})"
        roi_query = f"roi({title_escaped}, ROI)"

        # Determine if the movie is a success
        is_success = bool(list(prolog.query(success_query)))
        success_col.append(is_success)

        # Calculate the ROI of the movie
        roi_result = list(prolog.query(roi_query))
        if roi_result:
            roi_value = roi_result[0]['ROI']
            roi_col.append(round(roi_value, 2))
        else:
            roi_col.append(None)

    df['Success'] = success_col
    df['ROI'] = roi_col

    df.to_csv(output_csv_path, index=False)


# Execute the writing of the knowledge base
dataset_path = '../resources/dataset/Movie_dataset_cleaned.csv'
write_knowledge_base(dataset_path)

# Consult the knowledge base file
prolog = Prolog()
prolog.consult("kb.pl")

# Update the original dataset with Prolog results
update_dataset_with_prolog_results(dataset_path, '../resources/dataset/Movie_dataset_with_prolog_results.csv')
