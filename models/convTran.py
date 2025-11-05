import logging
import numpy as np

from .ConvTran.Models.model import model_factory, count_parameters
from .ConvTran.Models.optimizers import get_optimizer
from .ConvTran.Models.loss import get_loss_module
from .ConvTran.Training import SupervisedTrainer, train_runner


logger = logging.getLogger('__main__')

default_hyperparams = {
	'data_path': 'Dataset/UEA/', 'Norm': False,  'val_ratio': 0.25, 'print_interval': 10, 'Net_Type': ['C-T'],
	'emb_size': 12, 'dim_ff': 256, 'num_heads': 6,   'Fix_pos_encode': 'tAPE', 'Rel_pos_encode': 'eRPE',
	'epochs': 100,'batch_size': 16, 'lr': 0.001, 'dropout': 0.01, 'val_interval': 2, 'key_metric': 'accuracy',
	'gpu': 0,  'console': False, 'output_dir': 'Results/Dataset/UEA/',
}

def build_ConvTran_model(config,shape, n_labels, device="cuda", verbose=False):
	"""
	function to build the ConvTran model
	:param config: 		dict containing the hyperparameters
	:param shape: 		data shape
	:param n_labels: 	number of labels (classes)
	:param device: 		device to be used
	:param verbose: 	whether verbose output is required
	:return: 			UNtrained model
	"""
	if verbose:
		logger.info("Creating model ...")
	config['Data_shape'] = shape
	config['num_labels'] = n_labels

	model = model_factory(config)
	if verbose:
		logger.info("Model:\n{}".format(model))
		logger.info("Total number of parameters: {}".format(count_parameters(model)))
	# -------------------------------------------- Model Initialization ------------------------------------
	optim_class = get_optimizer("RAdam")
	config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
	config['loss_module'] = get_loss_module()
	model.to(device)

	return model



def build_train_ConvTran(train_loader, device, hyperparams,val_loader=None, save_path=None, verbose=False):
	"""
	function to build and train the ConvTran model
	:param train_loader: 	DataLoader for training
	:param device: 			device to train on
	:param hyperparams: 	dict of hyperparameters to be used during training
	:param val_loader: 		DataLoader for validation
	:param save_path: 		path where to save the model
	:param verbose: 		whether to have verbose output
	:return: 				epoch number where best val accuracy was obtained, model
	"""

	shape, n_labels = train_loader.dataset.feature.shape, np.unique(train_loader.dataset.labels).shape[0]

	model = build_ConvTran_model(hyperparams, shape , n_labels, device=device, verbose=verbose)
	# ---------------------------------------------- Validating The Model ------------------------------------
	if verbose:
		logger.info('Starting training...')

	# once get the SupervisedTrainer classes we can now train the model
	trainer = SupervisedTrainer(model, train_loader, device, hyperparams['loss_module'],
								hyperparams['optimizer'], l2_reg=0,print_interval=hyperparams['print_interval'],
								console=hyperparams['console'],print_conf_mat=False)

	val_evaluator = SupervisedTrainer(model, val_loader, device, hyperparams['loss_module'],
									  print_interval=hyperparams['print_interval'], console=hyperparams['console'],
		print_conf_mat=False) if val_loader is not None else None

	best_n_epochs, model = train_runner(hyperparams, model, trainer, save_path, val_evaluator=val_evaluator,
										verbose=verbose)

	return best_n_epochs, model