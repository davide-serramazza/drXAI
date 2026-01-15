from utils.trainers_aeon import train
from utils.data_utils import load_datasets
from utils.helpers import str2bool, elapsed_time

import argparse
import os
import pickle
import gc
import numpy as np

def main(args):
    base_path = args.dataset_dir
    saved_models_dir = args.saved_models_path
    model_names = args.classifiers
    results_file = args.result_file

    device = "cpu"  #HARDCODED

    results = {}

    for current_dataset in sorted(os.listdir(args.dataset_dir ) ):

        #results_path = os.path.join(results_dir, "_".join( (current_dataset ,"results") ) )+".npz"
        dataset_dir = os.path.join(base_path,current_dataset)
        data = load_datasets(dataset_dir, current_dataset)
        print("\n\n current loaded dataset is....", current_dataset)

        results[current_dataset] = {}

        for model_name in model_names:

            batch_size = -1
            # TODO use **kwargs to say which value is which param?
            model, current_accuracy, mem_used, training_time = elapsed_time(
                train,(data,  device, batch_size, model_name,False) )

            print(model_name,"training over! Accuracy is: ",current_accuracy, "\tTraining time:", training_time)
            file_name = "_".join((current_dataset,model_name,"allFeatures"))+".pkl"

            with open(os.path.join(saved_models_dir,file_name), 'wb') as f:
                pickle.dump(model, f)
            del model
            gc.collect()

            results[current_dataset][model_name] = {
                'accuracy' : current_accuracy,
                'average_memory_GB' : mem_used['average_memory_GB'],
                'peak_memory_GB' : mem_used['peak_memory_GB'],
                'training_time' : training_time
            }
            #torch.save(model, os.path.join(saved_models_dir,file_name))

    np.save(results_file, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="folder where datasets are stored")
    parser.add_argument("saved_models_path", type=str, help="folder where to save models")
    parser.add_argument("result_file", type=str, help=".npy file where to store results")
    parser.add_argument("--classifiers", nargs='+',help="classifier names")
    #parser.add_argument('--channel_selection',type=str2bool, default=False, help="whether to perform "
    #                                                                             "channel selection")
    #parser.add_argument('--time_point_selection',type=str2bool, default=False, help="whether to perform "
    #                                                                                "time point selection")
    #parser.add_argument('--n_samples',type=int, default=-1, help="how many instances to sample for each "
    #                                                             "class. Default is -1, meaning no sample")
    #TODO implement this!
    parser.add_argument("--initial_selection", type=str2bool, default=False)
    args = parser.parse_args()
    main(args)