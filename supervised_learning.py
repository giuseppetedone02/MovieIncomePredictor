## TO-DO
# Comprendere i modelli di regressione che utilizzerai (Regressione)
## Regressione Lineare
## Regressione Logistica
## Alberi di decisione (alberi di regressione)
## Random Forest (modello composito)
## Boosting/Functional Gradient Boosting (modello composito)
## Gradient-Boosted Trees (modello composito)
# Trovare migliori iper-parametri per i vari modelli di regressione
## K-fold cross validation
# Eseguire training e testing dei modelli di regressione
# Valutare i modelli di regressione (selezionando metriche piú appropriate)
## Discutere su possibile overfitting e underfitting e cause 
# ? Loss function per la regressione
# IMPORTANTE: Niente matrice di confusione

import csv
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, RandomForestRegressor

with open('./resources/Movie_dataset_filtered.csv', mode='r', encoding='utf-8-sig') as movieCsv:
    reader = csv.DictReader(movieCsv)
    data = list(reader)

def get_best_hyperparameters():
    
    dtr = DecisionTreeRegressor()
    rfr = RandomForestRegressor()
    lreg = LogisticRegression()
    
    seed = 15

    DecisionTreeHyperparameters = {
        'DecisionTree__criterion': [
            'squared_error', 'mse', 
            'friedman_mse', 'absolute_error', 
            'mae', 'poisson'
        ],
        'DecisionTree__splitter': ['best', 'random'],
        'DecisionTree__max_depth': [None, 10, 20, 40],
        'DecisionTree__min_samples_split': [2, 5, 10, 20],
        'DecisionTree__min_samples_leaf': [1, 2, 4, 8],
        'DecisionTree__max_features': ['auto', 'sqrt', 'log2'],
        'DecisionTree__random_state': [seed]
    }

    RandomForestHyperparameters = {
        'RandomForest__n_estimators': [100, 200, 400],
        'RandomForest__criterion': [
            'squared_error', 'absolute_error', 
            'friedman_mse', 'poisson'
        ],
        'RandomForest__max_depth': [None, 10, 20, 40],
        'RandomForest__max_features': ['auto', 'sqrt', 'log2'],
        'RandomForest_random_state': [seed]
    }

    LogisticRegressionHyperparameters = {
        'LogisticRegression__penalty': ['l1', 'l2', 'elasticnet'],
        'LogisticRegression_class_weight': [None, 'balanced'],
        'LogisticRegression__solver': [
            'newton-cholesky', 'newton-cg', 
            'lbfgs', 'sag', 'saga'
        ],
        'LogisticRegression__max_iter': [100, 400, 1000],
        'LogisticRegression__random_state': [seed]
    }