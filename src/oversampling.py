import pandas as pd
import numpy as np
import smogn

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.preprocessing import KBinsDiscretizer


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
        
        self.random_oversampler = RandomOverSampler(sampling_strategy=self.sampling_strategy, random_state=42, shrinkage=0)
        
        return self.random_oversampler.fit_resample(X, y_discretized)


def smogn_resample_data(X_train, y_train, targetColumn):
    train_data = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name=targetColumn)], axis=1)

    # Controllo preliminare sui dati
    if train_data.isnull().values.any():
        raise ValueError("I dati contengono valori nulli. Per favore, pulisci i dati prima di procedere.")

    train_data_resampled = smogn.smoter(data=train_data, y=targetColumn)
    X_train = train_data_resampled.drop(columns=[targetColumn]).to_numpy()
    y_train = train_data_resampled[targetColumn].to_numpy()

    return X_train, y_train

# Function for RandomOverSampler oversampling
# def random_oversampling(dataset, target, threshold=25):
#     def fit_resample(X, y):
#         dataset_copy = dataset.copy()
#         X = dataset_copy.drop(target, axis=1)
#         y = dataset_copy[target]

#         # Discretize the target variable into different bins
#         discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
#         dataset_copy[target] = discretizer.fit_transform(dataset_copy[[target]])

#         # Count the number of examples per class before oversampling
#         class_counts_before = dataset_copy[target].value_counts()
#         print("\nClass distribution before RandomOverSampling:")
#         print(class_counts_before)
        
#         # Apply RandomOverSampler
#         sampling_strategy = {cls: threshold for cls, count in class_counts_before.items() if count < threshold}
#         random_oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        
#         return random_oversampler.fit_resample(X, y)
#     return fit_resample
    
    # Create a new DataFrame with the resampled data
    # df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    # df_resampled[target] = y_resampled
    
    # # Count the number of examples per class after oversampling
    # class_counts_after = df_resampled[target].value_counts()
    # print("\nClass distribution after RandomOverSampling:")
    # print(class_counts_after)
    
    # return df_resampled


# Function for SMOTE oversampling
# def smote_oversampling(dataset, target, threshold=25):
#     dataset_copy = dataset.copy()
    
#     # Discretize the target variable into different bins
#     discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
#     dataset_copy[target] = discretizer.fit_transform(dataset_copy[[target]])
    
#     # Count the number of examples per class before oversampling
#     class_counts_before = dataset_copy[target].value_counts()
#     print("\nClass distribution before SMOTE oversampling:")
#     print(class_counts_before)

#     # Apply SMOTE oversampling
#     # sampling_strategy = {cls: threshold for cls, count in class_counts_before.items() if count < threshold}
#     smote = SMOTE(sampling_strategy="not majority", random_state=42, k_neighbors=5)

#     X = dataset_copy.drop(target, axis=1)
#     y = dataset_copy[target]
#     X_resampled, y_resampled = smote.fit_resample(X, y)
    
#     # Create a new DataFrame with the resampled data
#     df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
#     df_resampled[target] = y_resampled
    
#     # Count the number of examples per class after oversampling
#     class_counts_after = df_resampled[target].value_counts()
#     print("\nClass distribution after SMOTE oversampling:")
#     print(class_counts_after)
    
#     return df_resampled


# # Function for ADASYN oversampling
# def adasyn_oversampling(dataset, target):
#     dataset_copy = dataset.copy()
    
#     # Discretize the target variable into 10 bins
#     discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
#     dataset_copy[target] = discretizer.fit_transform(dataset_copy[[target]]).astype(int)
    
#     # Apply ADASYN oversampling
#     adasyn = ADASYN(sampling_strategy="not majority", random_state=42)
#     X = dataset_copy.drop(target, axis=1)
#     y = dataset_copy[target]
#     X_resampled, y_resampled = adasyn.fit_resample(X, y)
    
#     # Create a new DataFrame with the resampled data
#     df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
#     df_resampled[target] = y_resampled
    
#     return df_resampled
