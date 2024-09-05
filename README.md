# Movie Income Predictor
Progetto per l'esame di Ingegneria della Conoscenza presso l'Università degli Studi di Bari "Aldo Moro", anno 2023/2024.

## Ragionamento logico e Prolog
1. Installare il programma SWI-Prolog eseguendo il file swipl-9.2.0-1.x64.exe nella cartella requirements del progetto.
    - Durante l'installazione assicurarsi di spuntare la scelta di inserire la variabile d'ambiente di SWI-Prolog in PATH
2. Installare la libreria pyswip con il comando `pip install git+https://github.com/yuce/pyswip@master#egg=pyswip`
3. Se si intende eseguire il codice di ragionamento logico e Prolog, é sufficiente eseguire il seguente comando a partire dalla root del progetto:
`python ./src/movie_prolog.py`

## Esecuzione esperimenti
- Per poter eseguire il task di **apprendimento non supervisionato** é sufficiente eseguire il seguente comando a partire dalla root del progetto:
`python ./src/unsupervised_learning.py`

- Per poter eseguire i task di **apprendimento supervisionato** é sufficiente eseguire il seguente comando a partire dalla root del progetto:
`python ./src/supervised_learning.py`

- Per poter eseguire il task di **apprendimento probabilistico** é sufficiente eseguire il seguente comando a partire dalla root del progetto:
`python ./src/bayesian_network.py`

  **Nota:** É stata disabilitata l'esecuzione delle tecniche di over-sampling, in quanto la loro esecuzione richiederebbe molto tempo, oltre ad essere computazionalmente pesanti. Per abilitarne l'esecuzione, é sufficiente decommentare il contenuto di `pre_pipeline_oversampling` (righe 241-244 del file).

Si precisa che l'esecuzione dei task sopra menzionati causa la sovrascrittura della maggior parte dei file presenti nella cartella `resources` del progetto.
