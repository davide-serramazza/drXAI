import logging
import numpy as np
from copy import deepcopy

from torch.cuda import  empty_cache as empty_gpu_cache
from torch.utils.data import DataLoader

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


def build_train_ConvTran(train_loader,val_loader, dev_dataset, device, save_path=None, verbose=False):
	#tensorboard_writer = SummaryWriter('summary')
	# get basic info and build the initial model
	shape, n_labels = train_loader.dataset.feature.shape, np.unique(train_loader.dataset.labels).shape[0]

	model = build_ConvTran_model(default_hyperparams, shape , n_labels, device=device, verbose=verbose)
	# ---------------------------------------------- Validating The Model ------------------------------------
	if verbose:
		logger.info('Starting training...')

	# once get the SupervisedTrainer classes we can now train the model
	trainer = SupervisedTrainer(model, train_loader, device, default_hyperparams['loss_module'],
			default_hyperparams['optimizer'], l2_reg=0,print_interval=default_hyperparams['print_interval'],
				console=default_hyperparams['console'],print_conf_mat=False)

	val_evaluator = SupervisedTrainer(model, val_loader, device, default_hyperparams['loss_module'],
			print_interval=default_hyperparams['print_interval'], console=default_hyperparams['console'],
									  print_conf_mat=False)

	best_n_epochs, model = train_runner(default_hyperparams, model, trainer, save_path, val_evaluator=val_evaluator,
										verbose=verbose)

	# clean what used for validation
	del train_loader  ; del val_loader; del model
	empty_gpu_cache()
	# ---------------------------------------------- Final Training ------------------------------------
	# update hyper-parameters as the training set is now bigger
	final_default_hyperparams = deepcopy(default_hyperparams)
	final_default_hyperparams['epochs'] = best_n_epochs
	final_default_hyperparams['emb_size'] = np.ceil(default_hyperparams['emb_size'] / 0.75).astype(int)  # TODO hard coded
	final_default_hyperparams['num_heads'] = np.ceil(default_hyperparams['num_heads'] / 0.75).astype(int)

	# get final model, final trainer
	shape = dev_dataset.feature.shape
	final_model = build_ConvTran_model(final_default_hyperparams, shape, n_labels)
	dev_loader = DataLoader(dataset=dev_dataset, batch_size=trainer.dataloader.batch_size, shuffle=True, pin_memory=True)
	final_trainer = SupervisedTrainer(final_model, dev_loader, device, final_default_hyperparams['loss_module'], final_default_hyperparams['optimizer'],
			l2_reg=0, print_interval=final_default_hyperparams['print_interval'], console=final_default_hyperparams['console'],
									  print_conf_mat=False)

	# actually train the final model here
	_, final_model = train_runner(final_default_hyperparams, final_model, final_trainer, save_path, verbose=verbose)
	# get train set predictions on a NON shuffled dataloader and evaluate accuracy on test set

	dev_loader = DataLoader(dataset=dev_dataset, batch_size=trainer.dataloader.batch_size, shuffle=True, pin_memory=True)
	y_train_pred = final_model.predict(dev_loader)

	return final_model, y_train_pred, final_default_hyperparams