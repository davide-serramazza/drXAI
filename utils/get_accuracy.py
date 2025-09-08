import numpy as np
import timeit
import os
import torch
from copy import deepcopy

from  .helpers import extract_timePoints
from utils.trainers import trainer_dict

def get_accuracies(original_data,save_models_path, selections,clf_names, batch_sizes, initial_accuracies=None,channel_selection=True):

	for clf_name, batch_size in  zip(clf_names,batch_sizes):
		trainer = trainer_dict[clf_name]
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# get info
		current_dataset = original_data['name']
		accuracies = {	'accuracies'	: {}	}
		current_dataset_dict = accuracies['accuracies']


		# if the initial accuracy was provided initialize the dictionary using the 'all_channels' (aka initial) accuracy
		# otherwise the dictionary should be empty
		current_dataset_dict[clf_name] = {} if initial_accuracies is None or clf_name not in initial_accuracies.keys() else {
			'initial_accuracy' : initial_accuracies[clf_name]
		}

		for exp_name, selection in selections[clf_name].items():
			# accuracies vector
			current_dataset_accs = np.zeros(shape=(5,))

			# get current selected channels
			data  = deepcopy(original_data)
			if channel_selection:
				data['train_set']['X'] = data['train_set']['X'][:,selection,:]
				data['test_set']['X'] = data['test_set']['X'][:,selection,:]
			else:
				data['train_set']['X'] = extract_timePoints( data['train_set']['X'], selection )
				data['test_set']['X'] = extract_timePoints( data['test_set']['X'], selection )


			saved_models_path = os.path.join(save_models_path, "_".join((current_dataset,clf_name,exp_name))+".pth")
			# train 5 times
			for i in range(5):
				star_time = timeit.default_timer()
				current_accuracy , model = trainer(dataset=data, device=device, batch_size=batch_size)
				total_time = timeit.default_timer() - star_time
				current_dataset_accs[i] = current_accuracy

				# save best model
				if max(current_dataset_accs)==current_accuracy:
					torch.save(model,saved_models_path)
					training_time = total_time


			# extrac mean, std deviation and best accuracy
			current_dataset_dict[clf_name][exp_name]	 = 	{
				'training_time' : training_time,
				'mean' : np.mean(current_dataset_accs).item(),
				'std' : np.std(current_dataset_accs).item() ,
				'best' :  np.max(current_dataset_accs).item(),
				'channels' : selection
			}

	return current_dataset_dict