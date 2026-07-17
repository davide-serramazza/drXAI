#from utils.trainers_aeon import train
from utils.trainers import train
from utils.load_datasets import load_datasets
# TODO should I move some functions in other files?
from utils.helpers import *

import argparse
import os
import numpy as np
from copy import deepcopy


def main(args):
	# arguments extracting and processing
	base_path = args.dataset_dir
	saved_models_dir = args.saved_models_path

	# testing if classifier's list is in the allowed range
	model_names = args.classifiers
	all_clfs_allowed = np.all( [ m in ["HC2" ,"drCIF" ,"MRH" ,"ConvTran","hydra",""] for m in model_names ] )
	if all_clfs_allowed == False : raise ValueError("invalid classifier name(s)")

	results_file = args.result_file
	selection_dir = args.selection_dir
	channel_selection = args.channel_selection

	# data structure where results will be stored
	results = {}

	for f in sorted(os.listdir(args.dataset_dir ) ):

		# load data
		dataset_dir = os.path.join(base_path,f)
		original_data = load_datasets(dataset_dir, f)
		current_dataset_name = original_data['name']
		print("current loaded dataset is....", original_data['name'])

		if selection_dir:
			# if selection was provided, load the npy array then extract selections and relative names
			selection_file_names = [f for f in os.listdir(selection_dir) if (current_dataset_name in f and f.endswith(".npz")) ]
			selected_features_files = [ np.load(os.path.join(selection_dir,f),allow_pickle=True)['results'].item()
										for f in selection_file_names ]

			# initialize selection dictionary as empty dict
			selection_dict = {}
			for s in selected_features_files:
				selection_dict = get_computed_AI_selections(
					saliency_map_dict=s,channel_sel=channel_selection,
					selection_dict=selection_dict, info="")
		else:
			# train on all features
			selection_dict = {'allFeatures': {'allFeatures':None } }

		results[current_dataset_name] = {}

		for model_name in model_names:

			results[current_dataset_name][model_name] = {}

			for selection_name, selected_f in selection_dict.items():

				# extract features from original data if necessary
				data = original_data if selection_name=="allFeatures" else \
				extract_features(deepcopy(original_data) , selected_f,channel_selection)

				print("current evaluated selection is", selection_name, "of dataset", current_dataset_name)

				# prepare a dataset to save stats on the following training
				best_accuracy = -1
				story = {
					'accuracy' : [],
					'average_memory_GB' : [],
					'peak_memory_GB' :[],
					'training_time' : []
				}

				for i in range(3):
					print("training",(i+1),"-th model ...")
					model, current_accuracy, mem_used, training_time = elapsed_time(
						train, {
							'dataset':data,
							'model_name':model_name,
							'return_train_predictions':False
							}
					)

					story['accuracy'].append(current_accuracy)
					story['average_memory_GB'].append(mem_used['average_memory_GB'])
					story['peak_memory_GB'].append(mem_used['peak_memory_GB'])
					story['training_time'].append(training_time)

					print((i+1),")",model_name,"training over! Accuracy is: ",current_accuracy, "\tTraining time:", training_time)
					if current_accuracy > best_accuracy:
						current_accuracy = best_accuracy

						# save current best model
						file_name = "_".join((current_dataset_name, model_name, selection_name))
						save_model(file_name, model, model_name, saved_models_dir)

					# delete model and run garbage collector for memory tracking purposes
					clean_memory(model)

				# add current results to results data structure
				results[current_dataset_name][model_name][selection_name] = {
					'accuracy' : np.mean(story['accuracy']),
					'average_memory_GB' : np.mean(story['average_memory_GB']),
					'peak_memory_GB' : np.max(story['peak_memory_GB']),
					'training_time' : np.mean(story['training_time']),
					'story' : story,
				}
			np.save(results_file, results)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_dir", type=str, help="folder where datasets are stored")
	parser.add_argument("saved_models_path", type=str, help="folder where to save models")
	parser.add_argument("result_file", type=str, help=".npy file where to store results")
	parser.add_argument("--classifiers", nargs='+',help="classifier names")
	parser.add_argument("--selection_dir", type=str, default=None, help='feature selection(s) to be used. '
																		'If not provided all features are used.')
	parser.add_argument('--channel_selection',type=str2bool, default=False, help="whether to perform "
															"channel selection(True) or time point selection(False)")
	args = parser.parse_args()
	main(args)