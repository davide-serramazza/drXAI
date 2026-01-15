import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer

from sklearn.pipeline import Pipeline



class MultiRocketHydra():

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

        self.clf = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 10)
        )

        super().__init__()

    def fit(self,X,y):
        Xt_hydra  = self.hydra.fit_transform(X)
        Xt_multiRocket = self.multiRocket.fit_transform(X)

        Xt_total = np.concatenate([Xt_hydra,Xt_multiRocket],axis=1)

        self.clf.fit(Xt_total,y)

        return self

    def _predict(self,X) -> np.ndarray:
        Xt_hydra  = self.hydra.transform(X)
        Xt_multiRocket = self.multiRocket.transform(X)

        Xt_total = np.concatenate([Xt_hydra,Xt_multiRocket],axis=1)

        return self.clf.predict(Xt_total)

    def score(self,X,y):
        y_pred = self._predict(X)
        return accuracy_score(y,y_pred)

