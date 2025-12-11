import numpy as np
import os
import torch
from copy import deepcopy

from scipy.stats import hmean
from .helpers import elapsed_time, extract_timePoints

from .trainers import train

def get_accuracies(original_data,save_models_path, selections,clf_names, batch_sizes,channel_selection=True):

	for clf_name, batch_size in  zip(clf_names,batch_sizes):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# get info
		current_dataset = original_data['name']

		for name, selection_dict in selections[clf_name].items():

			if name=='initial accuracy':
				# skip cause this isn't a selection
				continue

			# accuracies vector
			selection = selection_dict['selection']
			n_orig_features = selection_dict['n_features']
			current_dataset_accs, current_dataset_hmeans = np.zeros(shape=(5,)),  np.zeros(shape=(5,))

			# get current selected channels
			data  = deepcopy(original_data)
			if channel_selection:
				data['train_set']['X'] = data['train_set']['X'][:,selection,:]
				data['test_set']['X'] = data['test_set']['X'][:,selection,:]
			else:
				data['train_set']['X'] = extract_timePoints( data['train_set']['X'], selection )
				data['test_set']['X'] = extract_timePoints( data['test_set']['X'], selection )


			saved_models_path = os.path.join(save_models_path, "_".join((current_dataset,clf_name,name))+".pth")

			# train 5 times measuring avg and std. dev. of metrics
			for i in range(5):
				current_accuracy , model, training_time = elapsed_time(
					train,(data,  device, batch_size, clf_name, False) )

				# saving current accuracy
				current_dataset_accs[i] = current_accuracy
				
				# computing and saving current hmean
				data_saved = 1 - len(selection)/n_orig_features
				current_dataset_hmeans[i] = hmean([data_saved,current_accuracy])

				# save best model
				if max(current_dataset_accs)==current_accuracy:
					torch.save(model,saved_models_path)


			# extrac mean, std deviation and best accuracy
			selections[clf_name][name]	 = 	{
				'training_time' : training_time,
                'selection' : selection,
                'accs' : {
                    'mean' : np.mean(current_dataset_accs).item(),
                    'std' : np.std(current_dataset_accs).item() ,
                    'best' :  np.max(current_dataset_accs).item(),
                },
                'hmeans' : {
                    'mean' : np.mean(current_dataset_hmeans).item(),
                    'std' : np.std(current_dataset_hmeans).item() ,
                    'best' :  np.max(current_dataset_hmeans).item(),
                }
			}

			print(clf_name, name, "evaluation computed!")

		print(clf_name,"evaluation over!")

	return selections
