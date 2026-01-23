import numpy as np

def class_prototypes_avg(X_train, y_train):

	num_classes = len(np.unique(y_train))
	prototypes = []
	for i in range(num_classes):
		current_samples = X_train[y_train == i]
		prototypes.append(np.mean(current_samples,axis=0))

	return np.mean(prototypes, axis=0, keepdims=True)