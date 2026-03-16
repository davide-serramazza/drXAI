import os
import h5py
import numpy as np

from aeon.datasets import load_from_ts_file
from sklearn.preprocessing import LabelEncoder


def load_datasets(dataset_dir, current_dataset ):

    # load data
    if current_dataset.count("monster")>0:
        # CASE FOR MONSTER DATASETS
        snapshot_id = os.listdir(os.path.join(dataset_dir,"snapshots"))[0]
        dataset_dir = os.path.join(dataset_dir,"snapshots",snapshot_id)

        print(f"Loading MONSTER dataset {current_dataset} from snapshot {snapshot_id}\n")

        X= np.load(os.path.join(dataset_dir, f"{current_dataset}_X.npy"))
        y = np.load(os.path.join(dataset_dir, f"{current_dataset}_y.npy"))
        # load the folds
        folds = [ np.loadtxt(os.path.join(dataset_dir,'test_indices_fold_'+str(i)+".txt")).astype(int) for i in range(5) ]
        # test set is the last fold,everything else is train set
        last_fold = ( X[folds[-1]]	,  y[folds[-1]] )
        first_folds = (np.concatenate( [ X[folds[i]] for i in range(4)] ) ,
                        np.concatenate( [ y[folds[i]] for i in range(4)] ) )

        current_dataset = current_dataset.split("--")[-1]

        if current_dataset=="MosquitoSound":
            X_train, y_train = last_fold;  X_test, y_test = first_folds
        else:
            X_train, y_train = first_folds;  X_test, y_test = last_fold

    elif current_dataset.endswith(".h5"):
        # case for synthetic .h5 files
        with h5py.File(dataset_dir, 'r') as f:
            X_train = f['train/X'][:]
            y_train = f['train/y'][:]
            X_test = f['test/X'][:]
            y_test = f['test/y'][:]
            current_dataset = current_dataset.replace(".h5","")

    elif current_dataset.endswith(".npy"):
        data = np.load(dataset_dir,allow_pickle=True ).item()
        X_train,  y_train = data['train']['X'], data['train']['y']
        X_test, y_test = data['test']['X'], data['test']['y']
        current_dataset = current_dataset.replace(".npy","")

    else :
        X_train, y_train = load_from_ts_file(os.path.join(dataset_dir, "_".join((current_dataset, "TRAIN.ts"  ))  ))
        X_test, y_test = load_from_ts_file(os.path.join(dataset_dir, "_".join((current_dataset, "TEST.ts"  ))  ))

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