import logging

from .ConvTran.Models.model import model_factory, count_parameters
from .ConvTran.Models.optimizers import get_optimizer
from .ConvTran.Models.loss import get_loss_module
from .ConvTran.Training import SupervisedTrainer, train_runner
from .ConvTran.Models.utils import load_model

logger = logging.getLogger('__main__')

default_hyperparams = {
	'data_path': 'Dataset/UEA/', 'Norm': False,  'val_ratio': 0.1, 'print_interval': 10, 'Net_Type': ['C-T'],
	'emb_size': 16, 'dim_ff': 256, 'num_heads': 8,   'Fix_pos_encode': 'tAPE', 'Rel_pos_encode': 'eRPE',
	'epochs': 100,'batch_size': 16, 'lr': 1e-3, 'dropout': 0.01, 'val_interval': 2, 'key_metric': 'accuracy',
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



def train_ConvTran(model,train_loader, device, hyperparams,val_loader=None, verbose=False):
	# TODO update documentation
	"""
	function to build and train the ConvTran model
	:param train_loader: 	DataLoader for training
	:param device: 			device to train on
	:param hyperparams: 	dict of hyperparameters to be used during training
	:param val_loader: 		DataLoader for validation
	:param verbose: 		whether to have verbose output
	:return: 				epoch number where best val accuracy was obtained, model
	"""

	if verbose:
		logger.info('Starting training...')

	# once get the SupervisedTrainer classes we can now train the model
	trainer = SupervisedTrainer(model, train_loader, device, hyperparams['loss_module'],
								hyperparams['optimizer'], l2_reg=0,print_interval=hyperparams['print_interval'],
								console=hyperparams['console'],print_conf_mat=False)

	val_evaluator = SupervisedTrainer(model, val_loader, device, hyperparams['loss_module'],
									  print_interval=hyperparams['print_interval'], console=hyperparams['console'],
		print_conf_mat=False) if val_loader is not None else None

	train_runner(hyperparams, model, trainer, val_evaluator=val_evaluator,
										verbose=verbose)
	# TODO keep path as constant?
	# TODO do i need optimizer here?
	# TODO do i also need start epoch?
	best_model, optimizer, start_epoch = load_model(model, "tmp/currentConvTran.pth", hyperparams['optimizer'])

	return best_model