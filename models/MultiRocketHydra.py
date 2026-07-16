import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from torch import Tensor

class MultiRocketHydra():
	"""
	implementation of MultiRocketHydra that
	 1) allow to select hyper-parameters for both MultiRocket and Hydra
	 2) uses RidgeClassifier's torch implementation that overcome limitation of sklearn's one
	"""

	def __init__(self,
			hydra_params = {},
			multiRocket_params = {},
			n_jobs = -1,
			batch_size = -1
		):

		self.batch_size = batch_size

		# initialize Hydra with the specified hyoer-params (if any) and its scaler
		self.hydra = HydraTransformer( n_jobs=n_jobs , **hydra_params)
		self.hydra_scaler = _SparseScaler()

		# initialize MultiRocket with the specified hyoer-params (if any) and its scaler as a pipeline
		self.multiRocket = Pipeline(
			steps=[('multiRocket',MultiRocket( n_jobs=n_jobs, **multiRocket_params)),
				   ('scaler',StandardScaler())]
		)

		# batch_size !=1 is meant for massive datasets. In this case use a
		# iterative solved for RidgeClassifier
		self.clf = RidgeClassifierCV (
			alphas=np.logspace(-3, 3, 10)
		) if self.batch_size == -1 else GridSearchCV (
			estimator=RidgeClassifier( solver='sparse_cg',max_iter=100,tol=0.001),
			param_grid={'alpha':np.logspace(-3, 3, 10)},cv=5,n_jobs=2
		)

		super().__init__()



	def _batched_hydra(self, X) -> Tensor:
		"""
		Performs a batched transformation of the input data using the hydra
		transform. This is useful for processing large datasets that cannot
		fit into memory in a single batch

		:param X: The input data to be transformed.
		:return: A tensor containing the transformed data after applying
		    the hydra transform in batches.
		:rtype: Tensor
		"""

		# init list container and compute n. batches
		Xt_hydra = []
		n_batches = int(np.ceil(X.shape[0] / self.batch_size))

		# batched transformations
		for i in range(n_batches):
			current_x = X[i * self.batch_size: min((i + 1) * self.batch_size, X.shape[0])]
			Xt_hydra.append(self.hydra.transform(current_x))

		# concatenate all results
		Xt_hydra = torch.cat(Xt_hydra, axis=0)
		return Xt_hydra


	def fit(self,X,y):
		"""

		:param X: data to classify
		:param y: labels
		:return:
		"""

		Xt_multiRocket = self.multiRocket.fit_transform(X)

		# hydra transform according to batch size
		if self.batch_size == -1:
			Xt_hydra  = self.hydra.fit_transform(X)
		else:
			# initialize hydra with one sample
			self.hydra.fit(X[ :1 , : , :])
			Xt_hydra = self._batched_hydra(X)

		Xt_hydra = self.hydra_scaler.fit_transform(Xt_hydra).numpy()


		Xt_total = np.concatenate([Xt_hydra,Xt_multiRocket],axis=1)

		self.clf = self.clf.fit(Xt_total,y)

		return self


	def _predict(self,X) -> np.ndarray:

		results = []

		# if  self.batch_size == -1 set it to X length
		if self.batch_size == -1:
			self.batch_size = X.shape[0]
		n_batches = int(np.ceil(X.shape[0] / self.batch_size))

		# in any case use a loop (will be only one iteration for batch_size==1
		for i in range(n_batches):
			current_x = X[i * self.batch_size: min((i + 1) * self.batch_size, X.shape[0])]

			Xt_hydra = self.hydra.transform(current_x)
			Xt_hydra = self.hydra_scaler.transform(Xt_hydra).numpy()

			Xt_multiRocket = self.multiRocket.transform(current_x)

			Xt_total = np.concatenate([Xt_hydra,Xt_multiRocket],axis=1)

			results.append(self.clf.predict(Xt_total))

		return np.concatenate(results)



	def score(self,X,y):
		"""
		score function that calls self.predict and computes accuracy
		:param X:
		:param y:
		:return: accuracy as a float
		"""

		y_pred = self._predict(X)
		return accuracy_score(y,y_pred)

