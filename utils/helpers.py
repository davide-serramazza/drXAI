import numpy as np
import random
import torch
import argparse

from .trainers import trainer_dict

def extract_timePoints( data, selection):
	new_data = []
	for intervals in sorted(selection):
		start, end = intervals.split(':')
		new_data.append(data [ :,:,int(start):int(end) ] )
	return np.concatenate(new_data,axis=-1)

def set_seed(seed: int = 42):
	"""Set random seeds for reproducibility"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

#TODO check at this accuracies argument
def get_computed_AI_selections(saliency_map_dict, channel_sel, selection_dict,  info):

	key2find = 'selected_channels_intersection' if channel_sel else 'selected_timePoints_intersection'

	for k in saliency_map_dict.keys():
		if k=='labels_map':
			continue

		if k=='accuracy':
			# prepare an entry in dict for current classifier
			model_name = info.replace("_","")
			init_acc = saliency_map_dict[k]
			if model_name in selection_dict.keys():
				# select only if in required classifiers!
				selection_dict[model_name] = { 'initial accuracy' :  init_acc}

		elif k==key2find:
			#k_name = k.replace(key2find,'')
			model_name, explainer = info.split("_")[1] , "_".join( info.split("_")[2:] )
			#for model in selection_dict.keys():
			if model_name in selection_dict.keys():
				# select only if in required classifiers!
				selection_dict[model_name][explainer] = { 'selection' : saliency_map_dict[k] }

		elif type(saliency_map_dict[k])==dict :
			get_computed_AI_selections(
				saliency_map_dict[k],channel_sel,selection_dict,
				info+"_"+str(k)	# update current info
			)

	return selection_dict


##################### functions to check arguments #####################

def extraction_method(channel_selection , time_point_selection):

	#TODO to check!!
	# only channel selection or time points can be selected
	assert channel_selection != time_point_selection, "Only channel selection or time points can be selected"
	print("performing channel selection") if channel_selection else print("performing time point selection")
	return channel_selection


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
		assert model_name in trainer_dict.keys(), "Classifier name not recognized"

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
