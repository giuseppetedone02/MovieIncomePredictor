# Movie Income Predictor
Progetto per l'esame di Ingegneria della Conoscenza presso l'Università degli Studi di Bari "Aldo Moro", anno 2023/2024. La documentazione del progetto é visibile consultando il file `Movie_Income_Predictor.pdf`.

## Ragionamento logico e Prolog
Per poter eseguire il codice di ragionamento logico e Prolog é sufficiente eseguire il seguente comando a partire dalla cartella `.\src` del progetto:
`python .\movie_prolog.py`

Prima di poter eseguire questo comando, é necessario rispettare alcuni requisiti:
1. Installare il programma SWI-Prolog (la versione del programma dev'essere superiore o uguale a 9.2.0);
    - Durante l'installazione assicurarsi di spuntare la scelta di inserire la variabile d'ambiente di SWI-Prolog in PATH
2. Installare la libreria `pyswip` con il comando `pip install git+https://github.com/yuce/pyswip@master#egg=pyswip`.

## Esecuzione esperimenti
- Per poter eseguire il task di **apprendimento non supervisionato** é sufficiente eseguire il seguente comando a partire dalla cartella `.\src` del progetto:
`python .\unsupervised_learning.py`

- Per poter eseguire i task di **apprendimento supervisionato** é sufficiente eseguire il seguente comando a partire dalla cartella `.\src` del progetto:
`python .\supervised_learning.py`

  **Nota:** É stata disabilitata l'esecuzione delle tecniche di over-sampling, in quanto la loro esecuzione richiederebbe molto tempo, oltre ad essere computazionalmente pesanti. Per abilitarne l'esecuzione, é sufficiente decommentare il contenuto di `pre_pipeline_oversampling` (righe 241-244 del file).

- Per poter eseguire il task di **apprendimento probabilistico** é sufficiente eseguire il seguente comando a partire dalla cartella `.\src` del progetto:
`python .\bayesian_network.py`

Si precisa che l'esecuzione dei task sopra menzionati causa la sovrascrittura della maggior parte dei file presenti nella cartella `.\resources` del progetto.
