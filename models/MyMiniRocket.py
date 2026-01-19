import torch.nn as nn

from .tsai.MINIROCKET_Pytorch import MiniRocketFeatures, get_minirocket_features

from .aaltd2024.code.ridge import RidgeClassifier

class MyMiniRocket(nn.Module):

	def __init__(self, n_channels,seq_len,n_classes,chunk_size=32, device="cpu", verbose=False):
		super(MyMiniRocket, self).__init__()

		self.n_channels = n_channels
		self.seq_len = seq_len
		self.n_classes = n_classes
		self.chunk_size = chunk_size
		self.device = device
		self.f_mean =None ; self.f_std = None

		self.transformer_model = MiniRocketFeatures(n_channels,seq_len,device=device)
		self.intermediate_dim = 9996

		self.classifier = RidgeClassifier(self.transformer_model,device=device)
		self.to(device)
		self.trained = False

	def train(self,data_train):

		self.classifier.fit(data_train)
		self.trained = True

	def score(self,data_test):
		error = self.classifier.score(data_test)
		acc = 1-error.cpu().numpy()

		return acc

	def predict(self,data):
		y = self.classifier.predict(data)
		return y

	def predict_proba(self,data):
		y = self.classifier.predict_proba(data)
		return y
