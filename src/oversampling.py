import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import KBinsDiscretizer


def random_oversampling(dataset, target):
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    dataset[target] = discretizer.fit_transform(dataset[[target]]).astype(int)
    
    ros = RandomOverSampler(sampling_strategy="not majority", random_state=42)

    X = dataset.drop(target, axis=1)
    y = dataset[target]

    # Fit the RandomOverSampler object to the dataset
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target] = y_resampled

    return df_resampled


def smote_oversampling(dataset, target):
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    dataset[target] = discretizer.fit_transform(dataset[[target]]).astype(int)
    
    smote = SMOTE(sampling_strategy="not majority", random_state=42, k_neighbors=5)

    X = dataset.drop(target, axis=1)
    y = dataset[target]

    # Fit the SMOTE object to the dataset
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target] = y_resampled

    return df_resampled