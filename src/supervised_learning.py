import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import smogn

from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from imblearn.pipeline import Pipeline

from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_validate, train_test_split, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
from imblearn.over_sampling import SMOTE, ADASYN

from oversampling import RandomOverSamplerTransformer, smogn_resample_data


def plot_learning_curves(
        regressionModel, X, y, regressionModelName, 
        logFile, oversamplingName):
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
    plt.title(regressionModelName + oversamplingName + ' Learning Curves')
    plt.savefig('../resources/plots/learning_curves/learning_curve_' + regressionModelName + oversamplingName + '.png')


def train_and_test_model(
        targetColumn, regressionModel, hyperParameters, 
        regressionModelName, seed, oversamplingName, pre_pipeline=[]):
    with open('../resources/logs/log_'+ regressionModelName + oversamplingName + ".txt", mode='w', encoding='utf-8-sig') as logFile:
        X = df.drop(columns=[targetColumn]).to_numpy()
        y = df[targetColumn].to_numpy()

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        # Apply SMOGN if specified in the pipeline
        if oversamplingName == '_SMOGN':
            X_train, y_train = smogn_resample_data(X_train, y_train, targetColumn)
            

        # Perform the grid search to find the best 
        # hyperparameters of the regression model
        pipeline = Pipeline(pre_pipeline + [(regressionModelName, regressionModel)])

        print('Iniziata la grid search con pipeline: ', pipeline)
        gridSearchCV = GridSearchCV(
            pipeline,
            param_grid = hyperParameters, 
            cv = cv, 
            n_jobs = -1,
            scoring = 'neg_mean_squared_error',
            error_score='raise'
        )
        print('===FINE GRID')
        gridSearchCV.fit(X_train, y_train)
        print('=================================================Fine Fit')

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
        plot_learning_curves(clf, X, y, regressionModelName, logFile, oversamplingName)


# Load the dataset
with open('../resources/dataset/Movie_dataset_features.csv', mode='r', encoding='utf-8-sig') as movieCsv:
    reader = csv.DictReader(movieCsv)
    dataset = list(reader)
    df = pd.DataFrame(dataset)

    targetColumn = 'Log_Worldwide_Gross'

    # Convert target column to numeric, coercing errors
    df[targetColumn] = pd.to_numeric(df[targetColumn], errors='coerce')

    seed = 42

# Define the cross-validation strategy and the scorer
cv = RepeatedKFold(n_splits=3, n_repeats=2, random_state=seed)


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


# Iterations with and without oversampling
ros_transformer = RandomOverSamplerTransformer()
smote = SMOTE(sampling_strategy="not majority", random_state=42, k_neighbors=5)

# Define the pre-pipeline oversampling strategies
pre_pipeline_oversampling = [
    [],
    [('SMOGN', None)],
    # [('RandomOverSamplerTransformer', ros_transformer), ('SMOTE', smote)],
]

models_and_hyperparameters = [
    (DecisionTreeRegressor(), DecisionTreeHyperparameters, 'DecisionTree'),
#     (RandomForestRegressor(), RandomForestHyperparameters, 'RandomForest'),
#     (LGBMRegressor(), LGBMRegressorHyperparameters, 'LGBMRegressor'),
#     (XGBRegressor(), XGBRegressorHyperparameters, 'XGBRegressor')
]

for pipe in pre_pipeline_oversampling:
    for regressionModel, hyperParameters, regressionModelName in models_and_hyperparameters:
        oversamplingName = '_' + pipe[-1][0] if len(pipe) > 0 else ''
        
        train_and_test_model(
            targetColumn, regressionModel, 
            hyperParameters, regressionModelName, 
            seed, oversamplingName, pre_pipeline=pipe
        )
