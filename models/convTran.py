import logging
import os

from .ConvTran.Models.model import model_factory, count_parameters
from .ConvTran.Models.optimizers import get_optimizer
from .ConvTran.Models.loss import get_loss_module
from .ConvTran.Training import SupervisedTrainer, train_runner
from .ConvTran.Models.utils import load_model

logger = logging.getLogger('__main__')

default_hyperparams = {
	'data_path': 'Dataset/UEA/', 'Norm': False,  'val_ratio': 0.1, 'print_interval': 10, 'Net_Type': ['C-T'],
	'emb_size': 16, 'dim_ff': 256, 'num_heads': 8,   'Fix_pos_encode': 'tAPE', 'Rel_pos_encode': 'eRPE',
	'epochs': 100,'batch_size': 256, 'lr': 1e-3, 'dropout': 0.01, 'val_interval': 2, 'key_metric': 'accuracy',
	'gpu': 0, 'early_stop_counter' :20, 'console': False, 'output_dir': 'Results/Dataset/UEA/',
}

def build_ConvTran_model(config,shape, n_labels, device="cuda", verbose=False):
	"""
	Builds a ConvTran model using the provided configuration and parameters.

	This function initializes a ConvTran model by setting up the given configuration,
	including data shape and the number of labels. It also configures the optimizer
	and loss module for the model. Finally, the model is moved to the specified computing
	device.

	:param config: 		Dictionary containing model configuration hyper-parameters.
	:param shape:	 	Tuple specifying the shape of the data to be processed by the model.
	:param n_labels: 	Integer representing the number of output labels for the model.
	:param device: 		Device on which the model will be executed. Defaults to "cuda".
	:param verbose: 	Boolean flag indicating whether to log detailed information
	    				during the model creation process. Defaults to False.
	:return: 			The initialized ConvTran model.
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



def train_ConvTran(model,train_loader, hyperparams,val_loader=None,  device='cuda', verbose=False):
	"""
	Trains the ConvTran model using the specified training and validation data, hyperparameters,
	and computing device. If a validation loader is provided, it evaluates the model periodically
	on validation data during training to monitor performance.

	:type model: 			Specific instance to be trained
	:param train_loader:	DataLoader object providing training data in batches.
	:param hyperparams: 	Dictionary containing training hyperparameter
	:param val_loader: 		DataLoader object providing validation data in batches.
	:param device: 			The computing device to use for training. Defaults is 'cuda'.
	:param verbose: 		Flag indicating whether to log detailed training progress and information.
	:return: 				The best model obtained during training loaded from the saved checkpoint file.
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

	i=0 ; tmp_file_name = "".join( ("tmp/currentConvTran", str(i) ,".pth") )
	while os.path.exists(tmp_file_name):
		i+=1 ;tmp_file_name = "".join( ("tmp/currentConvTran", str(i) ,".pth") )

	train_runner(hyperparams, model, trainer,tmp_file_name, val_evaluator=val_evaluator,verbose=verbose)

	best_model, optimizer, _ = load_model(model, tmp_file_name, hyperparams['optimizer'])

	os.remove(tmp_file_name)

	return best_model
