import os
import numpy as np

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

	# TODO: catch if is a monster dataset based on dataset_dir.count("monster")
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
		# case for UAE/'TimeSeriesClassification.com'
		# TODO is this necessary?
		print(f"Loading UAE dataset {current_dataset}\n")
		X_train, y_train = load_from_ts_file(os.path.join(dataset_dir, f"{current_dataset}_TRAIN.ts"))
		X_test, y_test = load_from_ts_file(os.path.join(dataset_dir, f"{current_dataset}_TEST.ts"))


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


def load_data_ConvTran(dataset , val_ratio=0.1,kwargs={}):

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
	splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_state) #, random_state=1234)
	train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(labels)), y=labels))

	train_data , train_label= data[train_indices], labels[train_indices]
	val_data, val_label = data[val_indices], labels[val_indices]

	return train_data, train_label, train_indices[0] , val_data, val_label, val_indices[0]


################################ DataLoader for different classifiers #######################################

def dataloader_hydra(dataset ,only_train=False,kwargs={}):
	# TODO only hydra case!
	# TODO can it be more tidy?
	X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']
	batch_size = kwargs['batch_size'] if kwargs.get('batch_size') else 256

	if not only_train:
		X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

	data_train =	Dataset(X_train, y_train, batch_size=batch_size, shuffle=False) if only_train else \
					Dataset(X_train, y_train, batch_size=batch_size, shuffle=True)

	# if only_train==False, return also the test set's DataLoader
	to_return = data_train if only_train else \
		(data_train,  Dataset(X_test, y_test, batch_size=batch_size, shuffle=False))

	return to_return


def dataloader_aeon(dataset, only_train=False,kwargs={}):

	data_train =  dataset['train_set']['X'] , dataset['train_set']['y']

	if not only_train:
		data_test  =        dataset['test_set']['X'], dataset['test_set']['y']

	# if only_train==False, return also the test set's
	to_return = data_train if only_train else (data_train,  data_test)

	return to_return