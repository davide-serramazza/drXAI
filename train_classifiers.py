# TODO it has to work with both trainers and trainers_aeon
#from utils.trainers_aeon import train
from utils.trainers import train
from utils.data_utils import load_datasets
from utils.helpers import str2bool, elapsed_time, save_model

import argparse
import os
import gc
import numpy as np

#TODO clean and comment

def main(args):
	base_path = args.dataset_dir
	saved_models_dir = args.saved_models_path

	# testing if classifier's list is in the allowed range
	# TODO can I be more specific?
	model_names = args.classifiers
	all_clfs_allowed = np.all( np.array(model_names) in ["HC2" ,"drCIF" ,"MRH" ,"ConvTran","hydra"] )
	if all_clfs_allowed == False : raise ValueError("invalid classifier names")

	results_file = args.result_file

	results = {}

	for f in sorted(os.listdir(args.dataset_dir ) ):

		dataset_dir = os.path.join(base_path,f)
		data = load_datasets(dataset_dir, f)
		print("\n\n current loaded dataset is....", data['name'])

		current_dataset = data['name']
		results[current_dataset] = {}

		for model_name in model_names:

			# TODO remove batch_size from here, use the one in model.ConvTran.py (default_hyperparams)
			best_accuracy = -1
			story = {
				'accuracy' : [],
				'average_memory_GB' : [],
				'peak_memory_GB' :[],
				'training_time' : []
			}

			for i in range(5):
				# TODO use **kwargs to say which value is which param?
				model, current_accuracy, mem_used, training_time = elapsed_time(
					train,(data, model_name,False) )

				story['accuracy'].append(current_accuracy)
				story['average_memory_GB'].append(mem_used['average_memory_GB'])
				story['peak_memory_GB'].append(mem_used['peak_memory_GB'])
				story['training_time'].append(training_time)

				print(i,")",model_name,"training over! Accuracy is: ",current_accuracy, "\tTraining time:", training_time)
				if current_accuracy > best_accuracy:
					current_accuracy = best_accuracy

					# save current best model
					file_name = "_".join((current_dataset,model_name,"allFeatures"))
					save_model(file_name, model, model_name, saved_models_dir)

				# delete model and run garbage collector for memory tracking purposes
				del model
				gc.collect()

			# add current results to results data structure
			results[current_dataset][model_name] = {
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
	parser.add_argument("--initial_selection", type=str2bool, default=False)
	args = parser.parse_args()
	main(args)