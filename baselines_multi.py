import pyximport
import numpy as np
import sys

pyximport.install(setup_args={
    'include_dirs': np.get_include(),
    'define_macros': [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
})


import pandas as pd
from pandas import DataFrame, Series
from tselect.channel_selectors.tselect import TSelect
from utils.load_datasets import load_datasets
import os
from numpy import savez_compressed
from aeon.transformations.collection.channel_selection import ElbowClassSum, ElbowClassPairwise
import timeit


data_dir = sys.argv[1]
save_dir = sys.argv[2]

def load_data(dir,f):
    # load data
    dataset_dir = os.path.join(dir, f)
    original_data = load_datasets(dataset_dir, f)
    current_dataset_name = original_data['name']
    print("\n\n current loaded dataset is....", original_data['name'])

    x_train, y_train = original_data['train_set']['X'], original_data['train_set']['y']
    x_test, y_test = original_data['test_set']['X'], original_data['test_set']['y']
    return x_train, y_train


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


    selection_dict['baselines'] = { 'noBG' : {
        "ECP" : {
            'selected_channels_intersection' : ecp.channels_selected_,
            'time' : elapsed_time_ecp
        },
        "ECS" : {'selected_channels_intersection' : ecs.channels_selected_,
                 'time' : elapsed_time_ecs
                 }
        }
    }



def get_tselect(selection_dict, x_train, y_train):
    start_time = timeit.default_timer()

    x_train, y_train = transform_data_tselect(x_train, y_train)

    elapsed_time = timeit.default_timer() - start_time

    channel_selector = TSelect()
    channel_selector.fit(x_train, y_train)
    selected = channel_selector.filtered_series
    elapsed_time = timeit.default_timer() - start_time

    selection_dict['baselines']['noBG']['TSelect'] = {
        'selected_channels_intersection': [int(s.split("_")[1]) for s in selected],
        "time" : elapsed_time

    }



for f in sorted(os.listdir(data_dir) ):

    print(f)
    x_train, y_train = load_data(data_dir,f)
    selection_dict = {}

    get_ELBOW_scores(selection_dict, x_train, y_train)
    print("elbow done")

    get_tselect(selection_dict, x_train, y_train)
    print("tselect done")

    file_name = f+"_EBLOW_TSelect_results.npz"
    savez_compressed( os.path.join( save_dir, file_name), results=selection_dict)