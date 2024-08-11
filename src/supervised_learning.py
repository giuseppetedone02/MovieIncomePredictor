## TO-DO
# Comprendere i modelli di regressione che utilizzerai
## LGBMRegressor (modello composito)
## Alberi di decisione (alberi di regressione)
## Random Forest (modello composito)
## Gradient-Boosted Trees (XGB, modello composito)
# Trovare migliori iper-parametri per i vari modelli di regressione
## K-fold cross validation
# Eseguire training e testing dei modelli di regressione
# Valutare i modelli di regressione (selezionando metriche piú appropriate)
## Discutere su possibile overfitting e underfitting e cause 
## Discutere su possibili soluzioni per overfitting e underfitting (Over/Under Sampling, Regularization, Dropout, Early Stopping, etc.)
# IMPORTANTE: Niente matrice di confusione

import csv
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate, train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score

with open('../resources/dataset/Movie_dataset_features.csv', mode='r', encoding='utf-8-sig') as movieCsv:
    reader = csv.DictReader(movieCsv)
    dataset = list(reader)
    df = pd.DataFrame(dataset)

    # Convert target column to numeric, coercing errors
    df['Log_BoxOffice'] = pd.to_numeric(df['Log_BoxOffice'], errors='coerce')

    X = df.drop(columns=['Log_BoxOffice']).to_numpy()
    y = df['Log_BoxOffice'].to_numpy()

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    seed = 42

# Define the cross-validation strategy and the scorer
cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=seed)

def plot_learning_curves(regressionModel, X, y, regressionModelName, logFile):
    # Calculate the learning curve for the given regression model
    train_sizes, train_scores, test_scores = learning_curve(
        regressionModel, 
        X, y, 
        cv=cv, 
        scoring='neg_mean_squared_error', 
        random_state=seed
    )

    # Write the learning curve to the log file
    logFile.write("\n{:<8}{:<25}{:<25}{:<25}{:<25}\n".format(
        "Size", 
        "Mean Train Score", 
        "Variance Train Score", 
        "Std Train Score", 
        "Train Scores"
    ))
    for i in range(0, len(train_scores)):
        mean = np.mean(train_scores[i])
        var = np.var(train_scores[i])
        std = np.std(train_scores[i])
        logFile.write("{:<8}{:<25}{:<25}{:<25}{:<25}\n".format(str(i), str(mean), str(var), str(std), str(train_scores[i])))
    
    logFile.write("\n{:<8}{:<25}{:<25}{:<25}{:<25}\n".format(
        "Size", 
        "Mean Test Score", 
        "Variance Test Score", 
        "Std Test Score", 
        "Test Scores"
    ))
    for i in range(0, len(test_scores)):
        mean = np.mean(test_scores[i])
        var = np.var(test_scores[i])
        std = np.std(test_scores[i])
        logFile.write("{:<8}{:<25}{:<25}{:<25}{:<25}\n".format(str(i), str(mean), str(var), str(std), str(test_scores[i])))
    
    # Plot the learning curve
    mean_train_errors = 1 - np.mean(train_scores, axis=1)
    mean_test_errors = 1 - np.mean(test_scores, axis=1)
    plt.figure()
    plt.plot(train_sizes, mean_train_errors, 'o-', color='r', label='Training Error')
    plt.plot(train_sizes, mean_test_errors, 'o-', color='g', label='Validation Error')
    plt.xlabel('Training examples')
    plt.ylabel('Mean Error')
    plt.legend(loc='best')
    plt.title(regressionModelName + ' Learning Curves')
    plt.savefig('../resources/plots/learning_curves/learning_curve_' + regressionModelName + '.png')


def train_and_test_model(regressionModel, hyperParameters, regressionModelName, seed=42):
    with open('../resources/logs/'+ regressionModelName + "_log.txt", mode='w', encoding='utf-8-sig') as logFile:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        

        # Perform the grid search to find the best 
        # hyperparameters of the regression model
        gridSearchCV = GridSearchCV(
            Pipeline([(regressionModelName, regressionModel)]), 
            param_grid = hyperParameters, 
            cv = cv, 
            n_jobs = -1,
            scoring = 'neg_mean_squared_error',
            error_score='raise'
        )
        gridSearchCV.fit(X_train, y_train)

        # Write the results to the log file
        logFile.write("Best parameters found:\n")
        for param, value in gridSearchCV.best_params_.items():
            logFile.write("{:<35}{:<10}\n".format(param, str(value)))        
        
        # Get the regression model with the best hyperparameters
        clf = gridSearchCV.best_estimator_

        # Evaluate the model
        metrics = {
            'MSE': make_scorer(mean_squared_error),
            'MAE': make_scorer(mean_absolute_error),
            'MSLE': make_scorer(mean_squared_log_error),
            'R2': make_scorer(r2_score)
        }
        scores = cross_validate(clf, X, y, cv=cv, scoring=metrics, n_jobs=-1)
        
        logFile.write("\n{:<10}{:<25}{:<25}{:<25}\n".format("Metric", "Score Mean", "Score Variance", "Score Std"))
        for key, scores in scores.items():
            if 'test_' in key:
                metric_name = key[5:]
                mean = np.mean(scores)
                var = np.var(scores)
                std = np.std(scores)

                logFile.write("{:<10}{:<25}{:<25}{:<25}\n".format(metric_name, str(mean), str(var), str(std)))

        # Plot the learning curve
        plot_learning_curves(clf, X, y, regressionModelName, logFile)


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
    'RandomForest__max_features': ['sqrt', 'log2'],
    'RandomForest__random_state': [seed]
}

LGBMRegressorHyperparameters = {
    'LGBMRegressor__n_estimators': [100, 200, 400],
    'LGBMRegressor__learning_rate': [0.01, 0.05, 0.1],
    'LGBMRegressor__max_depth': [5, 10, 20],
    'LGBMRegressor__num_leaves': [31, 127],
    'LGBMRegressor__class_weight': [None, 'balanced'],
    'LGBMRegressor__verbosity': [-1],
    'LGBMRegressor__random_state': [seed]
}

XGBRegressorHyperparameters = {
    'XGBRegressor__n_estimators': [100, 200, 400],
    'XGBRegressor__learning_rate': [0.01, 0.05, 0.1],
    'XGBRegressor__max_depth': [5, 10, 20],
    'XGBRegressor__num_leaves': [31, 127],
    'XGBRegressor__class_weight': [None, 'balanced'],
    'XGBRegressor__verbosity': [0],
    'XGBRegressor__random_state': [seed]
}

train_and_test_model(DecisionTreeRegressor(), DecisionTreeHyperparameters, 'DecisionTree', seed)
train_and_test_model(RandomForestRegressor(), RandomForestHyperparameters, 'RandomForest', seed)
train_and_test_model(LGBMRegressor(), LGBMRegressorHyperparameters, 'LGBMRegressor', seed)
train_and_test_model(XGBRegressor(), XGBRegressorHyperparameters, 'XGBRegressor', seed)