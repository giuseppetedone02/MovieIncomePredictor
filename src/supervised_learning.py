import csv
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score

with open('../resources/dataset/Movie_dataset_features.csv', mode='r', encoding='utf-8-sig') as movieCsv:
    reader = csv.DictReader(movieCsv)
    dataset = list(reader)
    df = pd.DataFrame(dataset)

    # Convert target column to numeric, coercing errors
    df['Log_Worldwide_Gross'] = pd.to_numeric(df['Log_Worldwide_Gross'], errors='coerce')

    X = df.drop(columns=['Log_Worldwide_Gross']).to_numpy()
    y = df['Log_Worldwide_Gross'].to_numpy()

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    seed = 42

# Define the cross-validation strategy and the scorer
cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=seed)
MSE_scorer = make_scorer(mean_squared_error)

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