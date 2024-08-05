import csv

def process_movie_dataset():
    with open('./resources/Movie_dataset.csv', mode='r', encoding='utf-8') as movieCsv:
        reader = csv.DictReader(movieCsv)
        print(reader)
        with open('kb.pl', mode='w', encoding='utf-8') as kbFile:
            for row in reader:
                print(row)
                kbFile.write('movie(' 
                + row['Title'] + ', ' + row['Year'] + ', ' + row['MPAA'] + ', ' 
                + row['Runtime'] + ', ' + row['Company'] + ', ' + row['Country'] + ', ' 
                + row['Director'] + ', ' + row['Writer'] + ', ' + row['Main Actor'] + ', ' 
                + row['Budget'] + ', ' + row['Worldwide Gross'] + ', ' 
                + row['Genre'] + ', ' + row['Score'] + ', ' + row['Votes'] 
                + ').\n')

if __name__ == "__main__":
    process_movie_dataset()