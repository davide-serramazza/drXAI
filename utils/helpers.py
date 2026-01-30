import numpy as np
import random
import torch
import argparse
import timeit
import os
import pickle
import gc

def extract_features(data, selection,channel_selection):
	if channel_selection:
		data['train_set']['X'] = data['train_set']['X'][:,selection,:]
		data['test_set']['X'] = data['test_set']['X'][:,selection,:]
	else:
		data['train_set']['X'], data['test_set']['X'] = extract_timePoints(data,selection)

	return data

def extract_timePoints( data, selection):
	# TODO documentation

	train_x, test_x = data['train_set']['X'], data['test_set']['X']
	new_data_train, new_data_test = [], []

	for intervals in sorted(selection):
		start, end = intervals.split(':')
		new_data_train.append(train_x [ :,:,int(start):int(end) ] )
		new_data_test.append(test_x [ :,:,int(start):int(end) ] )

	return np.concatenate(new_data_train,axis=-1), np.concatenate(new_data_test,axis=-1)

def set_seed(seed: int = 42):
	"""Set random seeds for reproducibility"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)


def save_model(file_name, model, model_name, saved_models_dir):
	if model_name in ["ConvTran","hydra","inceptionTime"]:
		torch.save(model, os.path.join(saved_models_dir,
									   "".join((file_name + ".pth"))
		))
	else:
		with open(os.path.join(saved_models_dir,
							   "".join((file_name + ".pkl"))), 'wb') as f:
			pickle.dump(model, f)


def get_computed_AI_selections(saliency_map_dict, channel_sel, selection_dict,  info):

	key2find = 'selected_channels_' if channel_sel else 'selected_timePoints_'
	main_key2find = key2find+'intersection'

	for k in saliency_map_dict.keys():
		if k=='labels_map':
			continue

	# TODO remove this comments
		# TDO to remove this???
		#if k=='accuracy':
			# prepare an entry in dict for current classifier
		#	model_name = info.replace("_","")
		#	init_acc = saliency_map_dict[k]
		#	if model_name in selection_dict.keys():
				# select only if in required classifiers!
		#		selection_dict[model_name]['initial accuracy']  =  init_acc

		elif k==main_key2find:
			#k_name = k.replace(key2find,'')
			_, model_name, background, explainer = info.split("_")

			# TODO extract a function here?
			if saliency_map_dict[main_key2find]!=[]:
				selection =saliency_map_dict[main_key2find]
			else:
				print(model_name,explainer,"intersection is empty!", end='!\t\t')
				k1 = key2find+ 'absoluteFirst'  ; k2 = key2find + 'averageFirst'
				selection = saliency_map_dict[k1] if saliency_map_dict[k1]!=None else saliency_map_dict[k2]
				print(k1,"as alternative")

			selection_dict[model_name]['_'.join((explainer,background))] =  selection


		elif type(saliency_map_dict[k])==dict :
			get_computed_AI_selections(
				saliency_map_dict[k],channel_sel,selection_dict,
				info+"_"+str(k)	# update current info
			)

	return selection_dict


def elapsed_time(f,args):
	"""
	record the running time of a function f
	:param f: function to be executed
	:param args: arguments to be passed to f
	:return: f's returned values, running time
	"""

	start_time = timeit.default_timer()
	returned_vales = f(**args)
	elapsed_time = timeit.default_timer() - start_time

	return *returned_vales, elapsed_time


def clean_memory(model):
	"""
	clear memory by deleting model and running garbage collect on both CPU and GPU
	:param model:
	:return:
	"""
	del model
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

##################### functions to check arguments #####################

def extraction_method(channel_selection , time_point_selection):

	assert channel_selection != time_point_selection, "Only channel selection or time points can be selected"
	print("performing channel selection") if channel_selection else print("performing time point selection")
	return channel_selection

classifiers_used = ['hydra','miniRocket','ConvTran']

def extract_classifiers_batchSizes(models_batchSizes: list[str]) -> tuple[list[str], list[int]]:
	"""
	extract models and relative batch_sizes
	:param models_batchSizes: list of classifiers followed by relative batch sizes
	:return:
	"""
	assert len(models_batchSizes)%2==0 , "batch size don't provide for any classifier"

	# extract models and batch sizes
	model_names = models_batchSizes[0::2]
	batch_sizes = [int(bs) for bs in models_batchSizes[1::2]]

	# check that provided classifiers are in the ones included in the study
	for model_name in model_names:
		assert model_name in classifiers_used, "Classifier name not recognized"

	return model_names, batch_sizes


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
