## TO-DO
# Comprendere i modelli di regressione che utilizzerai
## Regressione Logistica
## Alberi di decisione (alberi di regressione)
## Random Forest (modello composito)
## Gradient-Boosted Trees (modello composito)
# Trovare migliori iper-parametri per i vari modelli di regressione
## K-fold cross validation
# Eseguire training e testing dei modelli di regressione
# Valutare i modelli di regressione (selezionando metriche piú appropriate)
## Discutere su possibile overfitting e underfitting e cause 
# ? Loss function per la regressione
# IMPORTANTE: Niente matrice di confusione

import csv
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

with open('../resources/dataset/Movie_dataset_features.csv', mode='r', encoding='utf-8-sig') as movieCsv:
    reader = csv.DictReader(movieCsv)
    dataset = list(reader)
    df = pd.DataFrame(dataset)

    X = df.drop(columns=['Approx_Worldwide_Gross']).to_numpy()
    y = df['Approx_Worldwide_Gross'].to_numpy()

    seed = 42
    
def train_and_test_model(regressionModel, hyperParameters, regressionModelName, seed=42):
    with open('../resources/logs/'+ regressionModelName + "_log.txt", mode='w', encoding='utf-8-sig') as logFile:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Check the type of target data
        if not np.issubdtype(y_train.dtype, np.number):
            y_train = pd.to_numeric(y_train, errors='coerce')
        if not np.issubdtype(y_test.dtype, np.number):
            y_test = pd.to_numeric(y_test, errors='coerce')
        
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)

        gridSearchCV = GridSearchCV(
            Pipeline([(regressionModelName, regressionModel)]), 
            param_grid = hyperParameters, 
            cv = cv, 
            n_jobs = -1,
            scoring = scorer,
            error_score='raise'
        )
        gridSearchCV.fit(X_train, y_train)

        # Write the results to the log file
        logFile.write("Best parameters found:\n")
        for param, value in gridSearchCV.best_params_.items():
            logFile.write("{:<35}{:<10}\n".format(param, str(value)))        


DecisionTreeHyperparameters = {
    'DecisionTree__criterion': [
        'squared_error', 'friedman_mse', 
        'absolute_error', 'poisson'
    ],
    'DecisionTree__splitter': ['best', 'random'],
    'DecisionTree__max_depth': [None, 10, 20, 40],
    'DecisionTree__min_samples_split': [2, 5, 10, 20],
    'DecisionTree__min_samples_leaf': [1, 2, 4, 8],
    'DecisionTree__max_features': ['sqrt', 'log2'],
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

LGBMRegressorHyperparameters = {
    'LGBM__n_estimators': [100, 200, 400],
    'LGBM__learning_rate': [0.01, 0.05, 0.1],
    'LGBM__max_depth': [5, 10, 20],
    'LGBM__num_leaves': [31, 127],
    'LGBM__class_weight': [None, 'balanced'],
    'LGBM__verbosity': [-1],
    'LGBM__random_state': [seed]
}

train_and_test_model(DecisionTreeRegressor(), DecisionTreeHyperparameters, 'DecisionTree', seed)