import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer

from sklearn.pipeline import Pipeline

from models.aaltd2024.code.ridge import RidgeClassifier
from models.aaltd2024.code.utils import Dataset


#TODO clean and comment


class dummy_transform():
	""""
	dummy transform i.e. f(X) = X for RidgeClassifier
	"""
	def __init__(self, X):
		self.X = X
		self.num_features = X.shape[1]

	def __call__(self, *args, **kwargs):
		return args[0]


class MultiRocketHydra():
	"""
	implementation of MultiRocketHydra that
	 1) allow to select hyper-parameters for both MultiRocket and Hydra
	 2) uses RidgeClassifier's torch implementation that overcome limitation of sklearn's one
	"""

	def __init__(self,
			hydra_params = {},
			multiRocket_params = {},
			n_jobs = -1
		):
		self.hydra = Pipeline(
			steps=[('hydra',HydraTransformer( n_jobs=n_jobs , **hydra_params)),
				   ('scaler',_SparseScaler())]
		)
		self.multiRocket = Pipeline(
			steps=[('multiRocket',MultiRocket( n_jobs=n_jobs, **multiRocket_params)),
				   ('scaler',StandardScaler())]
		)

		self.clf = None	#temporally set to None

		super().__init__()

	def fit(self,X,y):
		"""

		:param X: data to classify
		:param y: labels
		:return:
		"""

		# transform data using both hydra and MultiRocket, then concatenate the two representations
		Xt_hydra  = self.hydra.fit_transform(X)
		Xt_multiRocket = self.multiRocket.fit_transform(X)
		Xt_total = np.concatenate([Xt_hydra,Xt_multiRocket],axis=1)

		# Instantiate torch's RidgeClassifier, create a data loader and finally fit the model
		self.clf = RidgeClassifier(dummy_transform(Xt_total))
		train_loader = Dataset(Xt_total,y,batch_size=X.shape[0])
		self.clf.fit(train_loader)

		return self

	def _predict(self,X,y) -> np.ndarray:

		# TODO understand this warning "/home/davide/miniconda3/envs/train_aeon_clfs/lib/python3.13/site-packages/sklearn/pipeline.py:61: FutureWarning: This Pipeline instance is not fitted yet. Call 'fit' with appropriate arguments before using other methods such as transform, predict, etc. This will raise an error in 1.8 instead of the current warning.
		Xt_hydra  = self.hydra.transform(X)
		Xt_multiRocket = self.multiRocket.transform(X)

		Xt_total = np.concatenate([Xt_hydra,Xt_multiRocket],axis=1)

		test_loader = Dataset(Xt_total,y,batch_size=X.shape[0],shuffle=False)

		return self.clf.predict(test_loader)


	def score(self,X,y):
		y_pred = self._predict(X,y)
		return accuracy_score(y,y_pred)

