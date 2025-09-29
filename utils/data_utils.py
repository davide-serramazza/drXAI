import os
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from aeon.datasets import load_from_ts_file
from models.ConvTran.utils import dataset_class


def sample_instances(X , y, n):
	"""
	sample instances from dataset (test set
	:param X:	samples
	:param y: 	labels
	:param n: 	number of instances to sample per class
	:return:
	"""

	# check the unique values in label set
	class_names, class_index, class_numbs = np.unique(y,return_inverse=True ,return_counts=True)

	# group instances by groups
	X_grouped = [ X[ np.where(class_index==i)[0] ] for i in range(class_numbs.shape[0]) ]
	y_grouped = [ y[ np.where(class_index==i)[0] ] for i in range(class_numbs.shape[0]) ]

	X = []
	y = []

	for current_class_instances, current_label in zip(X_grouped,y_grouped):

		assert current_class_instances.shape[0]==current_label.shape[0]
		# for each class sample up to n elements
		n_instances = current_class_instances.shape[0]
		selected = np.random.randint(low=0, high=n_instances,size=n) if n_instances>n else np.array([i for i in range(n_instances)])

		X.append(current_class_instances[selected]) ; y.append(current_label[selected])

	# from lists to arrays
	X = np.concatenate(X); y = np.concatenate(y)

	return X, y


def load_datasets(dataset_dir, current_dataset ):

	# load data

	X_train, y_train = load_from_ts_file(os.path.join(dataset_dir, f"{current_dataset}_TRAIN.ts"))
	X_test, y_test = load_from_ts_file(os.path.join(dataset_dir, f"{current_dataset}_TEST.ts"))

	# not sure if needed!
	#X_train , X_test = np.stack(X_train), np.stack(X_test)

	y_train, y_test,labels_map = to_numeric_labels(y_train, y_test)

	# data structure for dataset
	data = {
		'train_set': 		{},
		'test_set': 		{},
		'name' : 			current_dataset,
		'n_channels' : 		X_train.shape[1],
		'n_time_points_chunks':	20,	# hard coded!
	}

	# setting train, test sets and label map
	data['train_set']['X'] = X_train;	data['test_set']['X'] = X_test
	data['train_set']['y'] = y_train;	data['test_set']['y'] = y_test
	data['labels_map'] = labels_map



	return data


def to_numeric_labels(y_train, y_test):

	# convert labels to idx
	le = LabelEncoder()
	y_train = le.fit_transform( y_train)
	y_test = le.transform(y_test)

	return  y_train, y_test,  le.classes_


################################ ConvTran functions #######################################

def load_data_ConvTran(dataset , val_ratio=0.25, batch_size=32):

	# get different dataset parts
	X_train, y_train =      dataset['train_set']['X'] , dataset['train_set']['y']
	X_test, y_test =        dataset['test_set']['X'] , dataset['test_set']['y']

	# assuming equal length data
	_ , n_channels , seq_len = X_train.shape

	train_data, train_label, _, val_data, val_label, _ = split_dataset(X_train, y_train,val_ratio)

	# creating loaders
	train_dataset = dataset_class(train_data, train_label)
	val_dataset = dataset_class(val_data, val_label)
	dev_dataset = dataset_class( X_train, y_train)
	test_dataset = dataset_class(X_test,y_test)

	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
	val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

	return train_loader, val_loader, dev_dataset, test_loader



def split_dataset(data, label, validation_ratio, random_state = None):
	splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_state) #, random_state=1234)
	train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))

	train_data = data[train_indices]
	train_label = label[train_indices]
	val_data = data[val_indices]
	val_label = label[val_indices]

	return train_data, train_label, train_indices[0] , val_data, val_label, val_indices[0]


