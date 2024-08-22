import smogn
import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import KBinsDiscretizer


# Custom transformer to apply RandomOverSampler to a regression dataset
class RandomOverSamplerTransformer():
    def __init__(self, threshold=25):
        self.threshold = threshold

    def fit_resample(self, X, y):
        y = np.array(y).reshape(-1, 1)

        # Discretize the target variable into different bins
        discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
        y_discretized = discretizer.fit_transform(y).flatten()

        # Count the number of examples per class before oversampling
        occurrences = {item: list(y_discretized).count(item) for item in y_discretized}
        # print("\nClass distribution before RandomOverSampling:")
        # print(occurrences)
        
        # Apply RandomOverSampler
        self.sampling_strategy = {cls: self.threshold for cls, count in occurrences.items() if count < self.threshold}
        print("Sampling strategy:", self.sampling_strategy)
        
        self.random_oversampler = RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=42)
        
        return self.random_oversampler.fit_resample(X, y_discretized)


# Function to handle missing values in the dataset
def handle_missing_values(data):
    # Separare le feature numeriche e categoriche
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    # Imputazione mediana per le feature numeriche
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())

    # Imputazione moda per le feature categoriche
    for feature in categorical_features:
        data[feature] = data[feature].fillna(data[feature].mode()[0])

    return data


# Resample data with SMOGN oversampling (for supervised learning)
def smogn_resample_data(X_train, y_train, targetColumn):
    # Convert columns to numeric, if they aren't already
    X_train = pd.DataFrame(X_train)
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    y_train = pd.to_numeric(y_train, errors='coerce')
    
    train_data = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name=targetColumn)], axis=1)

    # Controllo preliminare sui dati
    if train_data.isnull().values.any():
        print("I dati contengono valori nulli. Pulizia dei dati in corso...")
        train_data = handle_missing_values(train_data)

    try:
        train_data_resampled = smogn.smoter(
            data=train_data, 
            y=targetColumn,
            samp_method='extreme'
        )
    except ValueError as e:
        if "synthetic data contains missing values" in str(e):
            print("Gestione dei valori mancanti nei dati sintetici...")
            train_data = handle_missing_values(train_data)
            train_data_resampled = smogn.smoter(data=train_data, y=targetColumn)
        else:
            raise

    X_train = train_data_resampled.drop(columns=[targetColumn]).to_numpy()
    y_train = train_data_resampled[targetColumn].to_numpy()

    return X_train, y_train
