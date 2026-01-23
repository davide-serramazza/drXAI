# TODO it has to work with both trainers and trainers_aeon
#from utils.trainers_aeon import train
from utils.trainers import train
from utils.data_utils import load_datasets
# TODO should I move some functions in other files?
from utils.helpers import elapsed_time, save_model, str2bool, get_computed_AI_selections, extract_timePoints

import argparse
import os
import gc
import numpy as np
from copy import deepcopy


# TODO move elsewhere
def extract_features(data, selection,channel_selection):
	if channel_selection:
		data['train_set']['X'] = data['train_set']['X'][:,selection,:]
		data['test_set']['X'] = data['test_set']['X'][:,selection,:]
	else:
		data['train_set']['X'], data['test_set']['X'] = extract_timePoints(data,selection)

	return data



def main(args):
	base_path = args.dataset_dir
	saved_models_dir = args.saved_models_path

	# testing if classifier's list is in the allowed range
	# TODO can I be more specific?
	model_names = args.classifiers
	all_clfs_allowed = np.all( np.array(model_names) in ["HC2" ,"drCIF" ,"MRH" ,"ConvTran","hydra"] )
	if all_clfs_allowed == False : raise ValueError("invalid classifier names")

	results_file = args.result_file
	selected_features = args.selection_dir
	channel_selection = args.channel_selection
	# TODO check selected features without channel selection

	results = {}

	for f in sorted(os.listdir(args.dataset_dir ) ):

		dataset_dir = os.path.join(base_path,f)
		original_data = load_datasets(dataset_dir, f)
		current_dataset_name = original_data['name']
		print("\n\n current loaded dataset is....", original_data['name'])

		if selected_features:
			selected_features_file = os.path.join(selected_features, current_dataset_name + "_results.npz")
			selections = get_computed_AI_selections(
				saliency_map_dict=np.load(selected_features_file,allow_pickle=True)['results'].item(),
				# TODO remove hydra level from selection_dict????
				channel_sel=channel_selection,selection_dict={'hydra':{}}, info="")
		else:
			selections = {'hydra': {'allFeatures':None }}


		results[current_dataset_name] = {}


		for model_name in model_names:

			results[current_dataset_name][model_name] = {}


			# TODO to remove hydra level from selection_dict????
			for selection_name, selected_f in selections['hydra'].items():

				data = original_data if selection_name=="allFeatures" else \
				extract_features(deepcopy(original_data) , selected_f,channel_selection)

				print("current evaluated selection is", selection_name)

				best_accuracy = -1
				story = {
					'accuracy' : [],
					'average_memory_GB' : [],
					'peak_memory_GB' :[],
					'training_time' : []
				}

				for i in range(5):
					print("training",(i+1),"-th model ...")
					# TODO use **kwargs to say which value is which param?
					model, current_accuracy, mem_used, training_time = elapsed_time(
						train,(data, model_name,False ) )

					story['accuracy'].append(current_accuracy)
					story['average_memory_GB'].append(mem_used['average_memory_GB'])
					story['peak_memory_GB'].append(mem_used['peak_memory_GB'])
					story['training_time'].append(training_time)

					print(i,")",model_name,"training over! Accuracy is: ",current_accuracy, "\tTraining time:", training_time)
					if current_accuracy > best_accuracy:
						current_accuracy = best_accuracy

						# save current best model
						file_name = "_".join((current_dataset_name, model_name, selection_name))
						save_model(file_name, model, model_name, saved_models_dir)

					# delete model and run garbage collector for memory tracking purposes
					del model
					gc.collect()

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
	#TODO implement this!
	parser.add_argument("--selection_dir", type=str, default=None, help='feature selection(s) to be used. '
																		'If not provided all features are used.')
	parser.add_argument('--channel_selection',type=str2bool, default=False, help="whether to perform "
															"channel selection(True) or time point selection(False)")
	args = parser.parse_args()
	main(args)