import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier, LogisticRegressionCV, RidgeClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from torch import Tensor


#from models.aaltd2024.code.ridge import RidgeClassifier
#from models.aaltd2024.code.utils import Dataset


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
		self.multiRocket = Pipeline(
			steps=[('multiRocket',MultiRocket( n_jobs=n_jobs, **multiRocket_params)),
				   ('scaler',StandardScaler())]
		)

		self.hydra = HydraTransformer( n_jobs=n_jobs , **hydra_params)
		self.hydra_scaler = _SparseScaler()


		self.clf = RidgeClassifierCV(
			alphas=np.logspace(-3, 3, 10)
		) if self.batch_size == -1 else GridSearchCV(
			estimator=RidgeClassifier(
				solver='sparse_cg',max_iter=200,tol=0.0005),
			param_grid={'alpha':np.logspace(-3, 3, 10)},cv=5,n_jobs=2)

		self.batch_size = batch_size

		super().__init__()


	def _batched_hydra(self, X) -> Tensor:
		Xt_hydra = []
		n_batches = int(np.ceil(X.shape[0] / self.batch_size))
		for i in range(n_batches):
			print(i,"out of",n_batches)
			current_x = X[i * self.batch_size: min((i + 1) * self.batch_size, X.shape[0])]
			Xt_hydra.append(self.hydra.transform(current_x))
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


	def _predict(self,X,y) -> np.ndarray:

		results = []

		if self.batch_size == -1:
			self.batch_size = X.shape[0]
		n_batches = int(np.ceil(X.shape[0] / self.batch_size))

		for i in range(n_batches):
			print(i,"out of",n_batches)
			current_x = X[i * self.batch_size: min((i + 1) * self.batch_size, X.shape[0])]

			Xt_hydra = self.hydra.transform(current_x)
			Xt_hydra = self.hydra_scaler.transform(Xt_hydra).numpy()

			Xt_multiRocket = self.multiRocket.transform(current_x)

			Xt_total = np.concatenate([Xt_hydra,Xt_multiRocket],axis=1)

			results.append(self.clf.predict(Xt_total))

		return np.concatenate(results)



	def score(self,X,y):
		y_pred = self._predict(X,y)
		return accuracy_score(y,y_pred)

