from copy import deepcopy
from torch.cuda import  empty_cache as empty_gpu_cache

from models.tsai.MINIROCKET_Pytorch import MiniRocketFeatures

from models.aaltd2024.code.hydra_gpu import HydraMultivariateGPU
from models.aaltd2024.code.ridge import RidgeClassifier
from models.aaltd2024.code.utils import *

from models.convTran import train_ConvTran, build_ConvTran_model, default_hyperparams as ConvTran_default_hyperparams

from utils.data_utils import load_data_ConvTran, dataloader_hydra_miniRocket


def _trainer_hydra_miniRocket( data_train, device, model_f):
    """
    function to train both hydra and miniRocket models
    :param data_train:  DataLoader for training set
    :param device:      device to train on
    :param model_f:     which model to use to transform the dataset
    :return:            trained model
    """

    # extract TS info
    _ , n_channels, length = data_train.shape
    n_classes =  data_train.classes.shape[0]

    # apply the specified transform function i.e.either 'HydraMultivariateGPU' or 'MiniRocketFeatures'
    transform = model_f(length,n_channels).to(device)

    # train GPU's RidgeClassifier
    model = RidgeClassifier(transform=transform, device=device)
    model.fit(data_train, num_classes=n_classes)

    return model


def _trainer_ConvTran( train_loader,val_loader, device ):
    # TODO do i need this method at all?
    # TODO update documentation
    """
    function to train ConvTran model
    :param train_loader:    DataLoader for training set (the remaining part after train-val split)
    :param val_loader:      DataLoader for validation set
    :param device:          device to train on
    :return:                trained model
    """

    shape, n_labels = train_loader.dataset.feature.shape, np.unique(train_loader.dataset.labels).shape[0]

    model = build_ConvTran_model(ConvTran_default_hyperparams, shape , n_labels, device=device, verbose=False)

    train_ConvTran(model,train_loader,device,ConvTran_default_hyperparams,val_loader,verbose=False)

    return model

def train(dataset, device, batch_size, model_name, return_train_predictions=True, verbose=False):
    """
    wrapper function to train each model included in the study
    :param dataset:                     current dataset
    :param device:                      device to train on
    :param batch_size:                  batch size to be used during training
    :param model_name:                  name of the model i.e. 'miniRocket' or 'hydra' or 'ConvTran'
    :param return_train_predictions:    whether to return train set predictions
    :param verbose:                     whether to have verbose output (only for ConvTran)
    :return:                            accuracy model and optionally train set predictions
    """

    # set functions (DataLoader, trainer, score) according to the current model

    dataloader_f = dataloader_hydra_miniRocket if model_name in ['hydra','miniRocket'] else load_data_ConvTran

    trainer_f = (lambda data, device: _trainer_hydra_miniRocket(data,device,HydraMultivariateGPU)) if model_name=='hydra' else \
        (lambda data, device: _trainer_hydra_miniRocket(data,device,MiniRocketFeatures)) if model_name=='miniRocket'\
            else lambda train_loader, val_loader, device : _trainer_ConvTran(train_loader,val_loader,device)

    score_f = (lambda model, data :(1- model.score(data).cpu().numpy().item()) ) if model_name in ['hydra','miniRocket'] \
        else (lambda model, data: model.eval().score(data))

    # use previously defined functions

    data_loader = dataloader_f(dataset, batch_size)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    model = trainer_f(*data_loader[:-1], device)

    mem_usage = {
        'peak_memory_GB': torch.cuda.memory_allocated()  / 1024**3,
        'average_memory_GB': torch.cuda.max_memory_allocated()  / 1024**3
    }

    accuracy = score_f( model, data_loader[-1] )

    to_return = model, accuracy, mem_usage

    if return_train_predictions:
        # if required, get train set predictions on a NON shuffled dataloader
        train_data = dataloader_f(dataset, batch_size,only_train=True)
        train_predictions = model.predict(train_data)
        to_return = (*to_return, train_predictions)

    return to_return