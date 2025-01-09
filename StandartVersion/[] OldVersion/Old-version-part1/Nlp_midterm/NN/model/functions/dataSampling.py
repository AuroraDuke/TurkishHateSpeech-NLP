# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:03:52 2024

@author: ilker
"""
def make_sampling(feature_union, X_train, X_test, y_train):
    import numpy as np
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    # Feature extraction
    X_train_features = feature_union.fit_transform(X_train).toarray()
    X_test_features = feature_union.transform(X_test).toarray()
    y_train = y_train.to_numpy()

    print("Initial X_train shape:", X_train_features.shape)

    # Compute dynamic thresholds based on combine_mean statistics
    combine_mean = np.mean(X_train_features, axis=1)
    threshold_low = np.percentile(combine_mean, 5)  # 5th percentile
    threshold_high = np.percentile(combine_mean, 95)  # 95th percentile

    # Apply dynamic thresholds
    mask = (combine_mean > threshold_low) & (combine_mean < threshold_high)
    X_train_features = X_train_features[mask]
    y_train = y_train[mask]

    print("Final X_train shape after applying thresholds:", X_train_features.shape)

    # Resampling strategies
    over_sampler = RandomOverSampler(random_state=42)
    under_sampler = RandomUnderSampler(random_state=42)
    smote = SMOTE(random_state=42, k_neighbors=10)

    samplings = {
        'Over-Sampling': over_sampler.fit_resample(X_train_features, y_train),
        'Under-Sampling': under_sampler.fit_resample(X_train_features, y_train),
        'SMOTE': smote.fit_resample(X_train_features, y_train)
    }

    return samplings, X_train_features, X_test_features, y_train
