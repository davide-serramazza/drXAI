import os
import numpy as np
import h5py

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from aeon.datasets import load_from_ts_file
from models.ConvTran.utils import dataset_class
from models.aaltd2024.code.utils import Dataset


#from datasets import load_dataset


def sample_instances(X , y_true, y_pred, n):
	"""
	sample instances from dataset (test set
	:param X:		samples
	:param y_true: 	true labels
	:param y_pred: 	predicted labels
	:param n: 		number of instances to sample per class among correct classified
	:return:
	"""

	# if n==-1 don't need to sample, so set n to number of total samples in the training set
	n = X.shape[0] if n==-1 else n

	# set y elements as either original idx (all >0 ) or -1 if misclassified
	y = np.array( [ y_true[i] if y_true[i] == y_pred[i] else -1 for i in range(y_true.shape[0])] )

	# check the unique values in labels set. Misclassified instances are
	# excluded since the corresponding entries were set to -1
	classes_idx = [np.where(y==class_id)[0] for class_id in np.unique(y_true)]

	X_sampled = []
	y_sampled = []
	sample_idx = []

	for current_class_idx in classes_idx:

		# for each class sample up to n elements
		#selected = np.random.randint(low=0, high=n_instances,size=n) if n_instances>n else np.array([i for i in range(n_instances)])
		current_n = min(n, len(current_class_idx))
		selected = np.random.permutation(current_class_idx)[:current_n]


		sample_idx.append(selected)
		X_sampled.append(X[selected]) ; y_sampled.append(y[selected])

	return  np.concatenate(X_sampled), np.concatenate(y_sampled), np.concatenate(sample_idx)


def load_datasets(dataset_dir, current_dataset ):

	# load data
	if current_dataset.count("monster")>0:
		# CASE FOR MONSTER DATASETS
		current_dataset = current_dataset.split("--")[-1]
		snapshot_id = os.listdir(os.path.join(dataset_dir,"snapshots"))[0]
		dataset_dir = os.path.join(dataset_dir,"snapshots",snapshot_id)

		print(f"Loading MONSTER dataset {current_dataset} from snapshot {snapshot_id}\n")

		X= np.load(os.path.join(dataset_dir, f"{current_dataset}_X.npy"))
		y = np.load(os.path.join(dataset_dir, f"{current_dataset}_y.npy"))
		# load the folds
		folds = [ np.loadtxt(os.path.join(dataset_dir,'test_indices_fold_'+str(i)+".txt")).astype(int) for i in range(5) ]
		# test set is the last fold,everything else is train set
		X_test = X[folds[-1]]	; y_test = y[folds[-1]]
		X_train = np.concatenate( [ X[folds[i]] for i in range(4)] ) ; y_train =  np.concatenate( [ y[folds[i]] for i in range(4)] )

	else:
		# case for synthetic .h5 files
		with h5py.File(dataset_dir, 'r') as f:
			X_train = f['train/X'][:]
			y_train = f['train/y'][:]
			X_test = f['test/X'][:]
			y_test = f['test/y'][:]
			current_dataset = current_dataset.replace(".h5","")

	y_train, y_test,labels_map = to_numeric_labels(y_train, y_test)


	# data structure for dataset
	data = {'train_set': {}, 'test_set': {}, 'name': current_dataset}

	# setting train, test sets and label map
	data['train_set']['X'] = X_train;	data['test_set']['X'] = X_test
	data['train_set']['y'] = y_train;	data['test_set']['y'] = y_test
	data['labels_map'] = labels_map

	print(f"Loaded {current_dataset} dataset with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples"
		  f" and {len(labels_map)} classes\n")

	return data


def to_numeric_labels(y_train, y_test):

	# convert labels to idx
	le = LabelEncoder()
	y_train = le.fit_transform( y_train)
	y_test = le.transform(y_test)

	return  y_train, y_test,  le.classes_


################################ ConvTran functions #######################################


def load_data_ConvTran(dataset , val_ratio=0.1,**kwargs):
	"""
	Loads data for ConvTran model

	:param dataset:		 A dictionary containing 'train_set' and 'test_set'.
	:param val_ratio: 	The ratio of the training dataset to be used for validation.	Defaults to 0.1.
	:param kwargs: 		Additional configuration arguments for the data loader, such as 'batch_size'.

	:return: tuple
	    A tuple containing three DataLoader objects:
	    - train_loader: DataLoader for the training data.
	    - val_loader: DataLoader for the validation data.
	    - test_loader: DataLoader for the testing data.
	"""

	# get different dataset parts
	X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']

	X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

	# assuming equal length data
	_ , n_channels , seq_len = X_train.shape

	train_data, train_label, _, val_data, val_label, _ = split_dataset(X_train, y_train,val_ratio)

	batch_size = kwargs['batch_size'] if kwargs.get('batch_size') else 256

	train_loader = DataLoader(dataset=dataset_class(train_data, train_label)
							  , batch_size=batch_size, shuffle=True, pin_memory=True)
	val_loader = DataLoader(dataset= dataset_class(val_data, val_label)
							, batch_size=batch_size, shuffle=False, pin_memory=True)
	test_loader = DataLoader(dataset=dataset_class(X_test,y_test)
							 , batch_size=batch_size, shuffle=False, pin_memory=True)

	return  train_loader, val_loader, test_loader



def split_dataset(data, labels, validation_ratio, random_state = None):
	"""
	Splits a dataset into training and validation subsets based on stratified sampling. This function
	is used in ConvTran training

	:param data: 				The dataset to be split. Expected to be an array-like object.
	:param labels: 				The labels associated with the dataset. Should have the same length as `data`.
	:param validation_ratio: 	The proportion of the dataset to be allocated to the validation subset.
	:param random_state: 		An optional seed or random state for reproducibility. If not provided, the
	    randomness will not be deterministic.

	:return: A tuple containing the following:
	    - train_data: 	The subset of the dataset intended for training.
	    - train_label: 	The labels corresponding to the training subset.
	    - train_indices: The indices in the original dataset corresponding to the training subset.
	    - val_data:		 The subset of the dataset intended for validation.
	    - val_label: 	The labels corresponding to the validation subset.
	    - val_indices: 	The indices in the original dataset corresponding to the validation subset.
	"""
	splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_state) #, random_state=1234)
	train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(labels)), y=labels))

	train_data , train_label= data[train_indices], labels[train_indices]
	val_data, val_label = data[val_indices], labels[val_indices]

	return train_data, train_label, train_indices[0] , val_data, val_label, val_indices[0]


################################ DataLoader for different classifiers #######################################

def dataloader_hydra(dataset ,only_train=False,**kwargs):
	"""
	Loads and processes data loaders for training and optionally testing datasets for hydra model

	:param dataset: 	A dictionary containing the training and testing datasets.
	:param only_train: A boolean flag to indicate whether to load only the training DataLoader.
	    If False, both training and testing DataLoaders are returned. Default is False.
	:param kwargs: Optional keyword arguments for DataLoader configuration.
	:return: Returns a DataLoader corresponding to the training set if only_train is True. If
	    only_train is False, returns a tuple of DataLoaders, corresponding to the training and
	    testing sets respectively.
	"""
	#get the batch size
	batch_size = kwargs['batch_size'] if kwargs.get('batch_size') else 256

	X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']
	X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

	data_train =	Dataset(X_train, y_train, batch_size=batch_size, shuffle=False) if only_train else \
		Dataset(X_train, y_train, batch_size=batch_size, shuffle=True)

	# if only_train==False, return also test set's DataLoader
	to_return = data_train if only_train else \
		(data_train,  Dataset(X_test, y_test, batch_size=batch_size, shuffle=False))

	return to_return


def dataloader_aeon(dataset,val_ratio= 0.0, **kwargs):

	data_train =  dataset['train_set']['X'] , dataset['train_set']['y']

	# split train set into train and val if necessary
	if val_ratio>0.0:
		train_data, train_label, train_indices , val_data, val_label, val_indices = split_dataset(
			data_train[0], data_train[1], val_ratio
		)
		data_train = train_data, train_label	; data_val = val_data, val_label

	data_test  =        dataset['test_set']['X'], dataset['test_set']['y']


	if val_ratio>0.0:
		to_return =  data_train, data_val,  data_test
	else:
		to_return = data_train, data_test

	return to_return