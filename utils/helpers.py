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

def get_computed_AI_selections(saliency_map_dict, selection_dict, accuracies, info, channel_sel):

	key2find = 'selected_channels_intersection' if channel_sel else 'selected_timePoints_intersection'
	for k in saliency_map_dict.keys():
		if k=='labels_map':
			continue

		if k=='accuracy':
			accuracies[info[1:]] = saliency_map_dict[k]
		elif k==key2find:
			#k_name = k.replace(key2find,'')
			model, explainer = info.split("_")[1] , "_".join( info.split("_")[2:] )
			#for model in selection_dict.keys():
			selection_dict[model][explainer] = saliency_map_dict[k]

		elif type(saliency_map_dict[k])==dict :
			get_computed_AI_selections(
				saliency_map_dict[k],selection_dict,accuracies,
				info+"_"+str(k), channel_sel)

	return selection_dict, accuracies


##################### functions to check arguments #####################

def extraction_method(channel_selection , time_point_selection):

	# only channel selection or time points can be selected
	assert channel_selection != time_point_selection, "Only channel selection or time points can be selected"
	print("performing channel selection") if channel_selection else print("performing time point selection")
	return channel_selection, time_point_selection


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
