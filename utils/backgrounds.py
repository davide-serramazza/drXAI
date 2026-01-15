import numpy as np

def class_prototypes_avg(X_train, y_train):

    num_classes = len(np.unique(y_train))
    prototypes = []
    for i in range(num_classes):
        current_samples = X_train[y_train == i]
        prototypes.append(np.mean(current_samples,axis=0))

    return np.mean(prototypes, axis=0, keepdims=True)

# TODO we don't need this!
def smote_avg(X_train, y_train):
    n_instance, n_channels, n_time_points = X_train.shape

    sm = SMOTE()
    resampled_X, resampled_y = sm.fit_resample( X_train.reshape((n_instance, -1)),  y_train)

    return resampled_X.reshape( -1, n_channels, n_time_points ).mean(axis=0, keepdims=True)