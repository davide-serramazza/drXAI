import numpy as np
import sys
import os
import pyximport
from aeon.datasets.tsc_datasets import univariate

pyximport.install(setup_args={
    'include_dirs': np.get_include(),
    'define_macros': [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
})

from scipy.signal import resample
import h5py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tselect.channel_selectors.tselect import TSelect
from utils.load_datasets import load_datasets
from numpy import savez_compressed
from aeon.transformations.collection.channel_selection import ElbowClassSum, ElbowClassPairwise
import timeit


data_dir = sys.argv[1]
save_dir = sys.argv[2]
type = sys.argv[3]
assert type in ["multi","uni"]

def load_data(dir,f):

    # load data
    dataset_dir = os.path.join(dir, f)
    original_data = load_datasets(dataset_dir, f)
    current_dataset_name = original_data['name']
    print("\n\n current loaded dataset is....", original_data['name'])

    x_train, y_train = original_data['train_set']['X'], original_data['train_set']['y']
    x_test, y_test = original_data['test_set']['X'], original_data['test_set']['y']
    return x_train, y_train, x_test, y_test, current_dataset_name




def transform_data_tselect(x_train, y_train):
    n, d, t = x_train.shape

    index = pd.MultiIndex.from_product(
        [range(n), range(t)],
        names=['sample', 'timestep']
    )

    # Reshape the array from (n, t, d) to (n*t, d) and create DataFrame
    x_train = pd.DataFrame(
        data=x_train.reshape(-1, d),  # Reshape to (n*t, d)
        index=index,  # MultiIndex (n, t)
        columns=[f'col_{i}' for i in range(d)]  # d columns
    )

    y_train = pd.Series(y_train)
    return x_train, y_train



########################## univariate baselines ##########################################

def get_ELBOW_scores(selection_dict, X_train, y_train):
    start_time = timeit.default_timer()

    # get elbow cut #
    ecp = ElbowClassPairwise()
    ecp.fit(X_train, y_train)
    elapsed_time_ecp = timeit.default_timer() - start_time

    start_time = timeit.default_timer()

    ecs = ElbowClassSum()
    ecs.fit(X_train, y_train)
    elapsed_time_ecs = timeit.default_timer() - start_time


    selection_dict['baselines']['noBG'] = {
        "ECP" : {
            'selected_channels_intersection' : ecp.channels_selected_,
            'time' : elapsed_time_ecp
        },
        "ECS" : {'selected_channels_intersection' : ecs.channels_selected_,
                 'time' : elapsed_time_ecs
                 }
    }



def get_tselect(selection_dict, x_train, y_train):
    start_time = timeit.default_timer()

    x_train, y_train = transform_data_tselect(x_train, y_train)

    channel_selector = TSelect()
    channel_selector.fit(x_train, y_train)
    selected = channel_selector.filtered_series
    elapsed_time = timeit.default_timer() - start_time

    selection_dict['baselines']['noBG']['TSelect'] = {
        'selected_channels_intersection': [int(s.split("_")[1]) for s in selected],
        "time" : elapsed_time

    }


######################### multivariate baselines ###############################


def get_random_forest_selection(selection_dict, x_train, y_train,ratios):

    start_time = timeit.default_timer()

    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(x_train.squeeze(), y_train)
    importances = np.flip( np.argsort(forest.feature_importances_) )

    elapsed_time = timeit.default_timer() - start_time

    for i,ratio in enumerate(ratios):
        n_to_select = int(len(importances)*ratio)

        selected = importances[-n_to_select:]

        selection_dict['baselines']['noBG']['random_forest_importance'+"_"+str(i)] = {
            'selected_channels_intersection': [ ":".join( (str(s), str(s+1)) ) for s in  selected],
            "time" : elapsed_time
        }


def get_resampled_data(X_train,X_test,y_train,y_test, ratios,selection_dict,file_name,save_dir):

    start_time = timeit.default_timer()

    orig_length = X_train.shape[2]

    for i,ratio in enumerate(ratios):
        dowsampled_X_train = resample(X_train, int(ratio*orig_length),axis=-1)
        dowsampled_X_test = resample(X_test, int(ratio*orig_length),axis=-1)

        elapsed_time = timeit.default_timer() - start_time

        selection_dict['baselines']['noBG'] [ "_".join ( ('downsampled',str(i)) )] = {
            "time" : elapsed_time
        }

        new_file_name = "_".join( (file_name,"downsampled",str(i)) )
        with h5py.File(os.path.join(save_dir, new_file_name+".h5" ), 'w') as f:
            train = f.create_group('train')
            train.create_dataset('X', data=dowsampled_X_train,compression='gzip')
            train.create_dataset('y', data=y_train)

            test = f.create_group('test')
            test.create_dataset('X', data=dowsampled_X_test,compression='gzip')
            test.create_dataset('y', data=y_test)


    return X_train, X_test



univariate_ratios = {
    'RightWhaleCalls' : [0.4,	0.15],
    'CornellWhaleChallenge' : [0.3,	0.1],
    'MosquitoSound' : [0.1 , 0.15],
    'WhaleSounds' : [ 0.1 ]
}


for f in sorted(os.listdir(data_dir) )[-1:]:

    print(f)
    x_train, y_train, x_test, y_test, current_dataset_name = load_data(data_dir,f)

    selection_dict = { 'baselines' : {
        'noBG' : {}
    } }

    if type=="multi":
        get_ELBOW_scores(selection_dict, x_train, y_train)
        print("elbow done")

        get_tselect(selection_dict, x_train, y_train)
        print("tselect done")

        file_name = f+"_EBLOW_TSelect_results.npz"
    elif type=="uni":
        get_random_forest_selection(selection_dict, x_train, y_train, univariate_ratios[current_dataset_name])
        print("random forest done")

        get_resampled_data(x_train,x_train,y_train,y_train,univariate_ratios[current_dataset_name]
                           ,selection_dict,current_dataset_name,save_dir)

        file_name = f+"_rfImportance_downsampling.npz"

    else:
        raise ValueError("type must be either multi or uni")
    savez_compressed( os.path.join( save_dir, file_name), results=selection_dict)
