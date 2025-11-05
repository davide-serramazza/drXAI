from copy import deepcopy
from torch.cuda import  empty_cache as empty_gpu_cache

from models.tsai.MINIROCKET_Pytorch import MiniRocketFeatures
from models.aaltd2024.code.hydra_gpu import HydraMultivariateGPU
from models.aaltd2024.code.ridge import RidgeClassifier
from models.aaltd2024.code.utils import *
from models.convTran import build_train_ConvTran
from models.convTran import default_hyperparams as ConvTran_default_hyperparams
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


def _trainer_ConvTran( train_loader,val_loader,dev_loader, device ):
    """
    function to train ConvTran model
    :param train_loader:    DataLoader for training set (the remaining part after train-val split)
    :param val_loader:      DataLoader for validation set
    :param dev_loader:      DataLoader for development set i.e. the whole training set
    :param device:          device to train on
    :return:                trained model
    """

    # validation stage
    best_n_epochs, _ =  build_train_ConvTran(train_loader,
                                    val_loader=val_loader,device=device,hyperparams=ConvTran_default_hyperparams)
    empty_gpu_cache()

    # increase the emb size (and consequently the number of heads) by 25% as the training data are
    # increasing by the same amount
    final_hyperparams = deepcopy(ConvTran_default_hyperparams)
    final_hyperparams['epochs'] = best_n_epochs
    final_hyperparams['emb_size'] = np.ceil(ConvTran_default_hyperparams['emb_size'] / 0.75).astype(int)  # TODO hard coded
    final_hyperparams['num_heads'] = np.ceil(ConvTran_default_hyperparams['num_heads'] / 0.75).astype(int)

    # train the final model WITHOUT any validation set!
    _ , model = build_train_ConvTran(dev_loader,val_loader=None, device=device,hyperparams=final_hyperparams)

    return model



def train(dataset, device, batch_size, model_name, return_train_predictions=False, verbose=False):
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
            else _trainer_ConvTran

    score_f = (lambda model, data :(1- model.score(data).cpu().numpy().item()) ) if model_name in ['hydra','miniRocket'] \
        else (lambda model, data: model.eval().score(data))

    # use previously defined functions

    data_loader = dataloader_f(dataset, batch_size)

    model = trainer_f(*data_loader[:-1], device)

    accuracy = score_f( model, data_loader[-1] )

    to_return = accuracy, model

    if return_train_predictions:
        # if required, get train set predictions on a NON shuffled dataloader
        train_data = dataloader_f(dataset, batch_size,only_train=True)
        train_predictions = model.predict(train_data)
        to_return = (*to_return, train_predictions)

    return to_return