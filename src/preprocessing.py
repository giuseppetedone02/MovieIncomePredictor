import csv
import pandas as pd


def process_movie_dataset():
    with open('../resources/dataset/Movie_dataset_cleaned.csv', mode='r', encoding='utf-8-sig') as movieCsv:
        reader = csv.DictReader(movieCsv)

        # Write the Prolog knowledge base to a file
        with open('kb.pl', mode='w', encoding='utf-8') as kbFile:
            for row in reader:
                # Write a Prolog fact for each row in the CSV file
                # using 'repr' to escape special characters
                kbFile.write('movie(' 
                + repr(row['Title']) + ', ' + row['Year'] + ', ' + repr(row['MPAA']) + ', ' 
                + row['Runtime'] + ', ' + repr(row['Company']) + ', ' + repr(row['Country']) + ', ' 
                + repr(row['Director']) + ', ' + repr(row['Writer']) + ', ' + repr(row['Main Actor']) + ', ' 
                + row['Budget'] + ', ' + row['Worldwide Gross'] + ', ' 
                + repr(row['Genre']) + ', ' + row['Score'] + ', ' + row['Votes'] 
                + ').\n')

def remove_na_from_movie_dataset(input_csv_path, output_csv_path):
    # Read the dataset from the CSV file, including malformed rows
    df = pd.read_csv(input_csv_path, on_bad_lines='skip')
    
    # Remove rows that have empty values in any column
    df_cleaned = df.dropna()
    
    # Save the cleaned dataset to a new CSV file
    df_cleaned.to_csv(output_csv_path, index=False)

process_movie_dataset()