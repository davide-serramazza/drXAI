from models.MultiRocketHydra import MultiRocketHydra

from models.aaltd2024.code.hydra_gpu import HydraMultivariateGPU
from models.aaltd2024.code.ridge import RidgeClassifier
from models.aaltd2024.code.utils import *

from models.convTran import train_ConvTran, build_ConvTran_model, default_hyperparams as ConvTran_default_hyperparams

from models.inceptionTime.inception_time_pytorch.model import InceptionTime

from utils.data_utils import load_data_ConvTran, dataloader_hydra, dataloader_aeon

from memory_profiler import  memory_usage

exceptions = {
    ('MRH', 'MosquitoSound')  : { 'batch_size':24000, 'multiRocket_params' : {'n_kernels' : 3125}},
    ('MRH', 'WhaleSounds') : { 'batch_size':24000, 'multiRocket_params' : {'n_kernels' : 3125}},


    ('ConvTran' , 'arc_loss') : {  'batch_size' : 64},
    ('ConvTran' ,'RightWhaleCalls') : {  'batch_size' : 8,'early_stop_counter':10},
    ('ConvTran' , 'CornellWhaleChallenge')  : {  'batch_size' : 8,'early_stop_counter':10},
    ('ConvTran' , 'MosquitoSound') : {  'batch_size' : 12,'early_stop_counter':10},
    ('ConvTran', 'WhaleSounds')  : { 'batch_size':16,'early_stop_counter':10},
    ('ConvTran' , 'UrbanSound') : {'batch_size':1},
    ('ConvTran', 'syntheticUni')  : {  'batch_size' : 1, 'early_stop_counter':10},


    ('inceptionTime' , 'arc_loss') : {  'batch_size' : 128},
    ('inceptionTime' ,'RightWhaleCalls') : { 'early_stop_counter':10},
    ('inceptionTime' , 'CornellWhaleChallenge')  : {  'early_stop_counter':10},
    ('inceptionTime' , 'MosquitoSound') : { 'early_stop_counter':10},
    ('inceptionTime', 'WhaleSounds')  : { 'early_stop_counter':10},
    ('inceptionTime', 'syntheticUni')  : {  'batch_size' : 60, 'early_stop_counter':10},
}


def profile_function(func, *args, **kwargs):
    """
	profile a function's memory usage and runtime
	:param func: 	function to be profiled
	:param args: 	arguments to be passed to func
	:param kwargs: 	keyword arguments to be passed to func
	:return: function's result i.e. trained model and memory usage statistics
	"""

    # Variable to store the result
    trained_model = []

    def wrapper():
        """
		wrapper that executes func with the right arguments and store the result
		:return:
		"""
        """Wrapper that captures the return value"""
        result = func(*args, **kwargs)
        trained_model.append(result)

    # Get memory usage and runtime statistics
    mem_usage = memory_usage(
        (wrapper, (), {}),
        interval=0.2,
        timeout=1,
        include_children=True
    )

    # Calculate memory statistics
    mem_usage = {
        'peak_memory_GB': max(mem_usage) / 1024,
        'average_memory_GB': sum(mem_usage) / len(mem_usage) / 1024,
    }

    return trained_model[0], mem_usage



def _train_aeon(data,model):
    X_train, y_train = data
    model.fit(X_train, y_train)

    return model

def _trainer_hydra( data_train, device="cuda"):
    """
    function to train hydra model

    :param data_train:  DataLoader for training set
    :param device:      device to train on
    :return:            trained model
    """

    # extract TS info
    _ , n_channels, length = data_train.shape
    n_classes =  data_train.classes.shape[0]

    # apply the specified transform function i.e.either 'HydraMultivariateGPU' or 'MiniRocketFeatures'
    transform = HydraMultivariateGPU(length,n_channels).to(device) #model_f(length,n_channels).to(device)

    # train GPU's RidgeClassifier
    model = RidgeClassifier(transform=transform, device=device)
    model.fit(data_train, num_classes=n_classes)

    return model


def _trainer_ConvTran( train_loader,val_loader,  **kwargs ):
    """
    Train a ConvTran model using the provided training and validation dataloaders, with
    optional parameter overrides.

    The method configures a ConvTran model based on the default hyperparameters, updates the
    parameters with the values provided in the `kwargs` dictionary, and trains the model
    using the given dataloaders. The trained model is then returned for further use or
    evaluation.

    :param train_loader: The dataloader for the training dataset.
    :param val_loader: The dataloader for the validation dataset.
    :param kwargs: Optional dictionary containing parameter overrides. These parameters
        modify the default ConvTran hyperparameters to customize the model's architecture
        or training behavior
    :return: The trained ConvTran model instance
    """


    for k in kwargs: ConvTran_default_hyperparams[k] = kwargs[k]

    shape, n_labels = train_loader.dataset.feature.shape, np.unique(train_loader.dataset.labels).shape[0]

    model = build_ConvTran_model(ConvTran_default_hyperparams, shape , n_labels, verbose=False)

    train_ConvTran(model,train_loader,ConvTran_default_hyperparams,val_loader,verbose=False)

    return model

def _train_inceptionTime(train_data, val_data, **kwargs):
    """
    Trains an InceptionTime model using the given training and validation data. The method initializes
    the model with default parameters, unless overridden via keyword arguments.

    :param train_data: The training dataset used to train the InceptionTime model.
    :param val_data: The validation dataset used to validate the InceptionTime model.
    :param kwargs: Optional keyword arguments that can be used to customize model parameters:
    :return: A trained instance of the InceptionTime model.
    """
    batch_size = 256 if 'batch_size' not in kwargs else kwargs['batch_size']
    early_stop_counter=20 if 'early_stop_counter' not in kwargs else kwargs['early_stop_counter']
    model = InceptionTime(train_data, val_data, batch_size=batch_size, filters=32, depth=6, models=5,
                          early_stop_counter=early_stop_counter)
    model.fit(learning_rate=1e-3,epochs=100,verbose=False)

    return model


def train(dataset, model_name, return_train_predictions=False):
    """
    Trains a machine learning model specified by the configuration, evaluates its performance, and optionally returns
    predictions on the training dataset.

    :param dataset:                 The dataset containing training and validation data
    :param model_name:              Name of the machine learning model to train.
    :param return_train_predictions: If True, predictions for the training dataset will be returned along with the trained
        model, accuracy, and memory usage statistics. Defaults to False.

    :return: A tuple containing the trained model, validation accuracy, memory usage statistics (peak and average memory
        usage in GB), and optionally training data predictions if `return_train_predictions` is True.
    """

    # get optional hyper parameters for specific model/dataset combinations
    key = (model_name, dataset['name'].split('_downsampled_')[0] )
    hyper_params = exceptions[key] if key in exceptions else {}

    # set data loaders, trainer and score functions according to current model
    dataloader_f = {
        'ConvTran' : load_data_ConvTran,
        'hydra' : dataloader_hydra,
        'MRH' : dataloader_aeon,
        'inceptionTime' : lambda data,**kwargs: dataloader_aeon(data,val_ratio=0.1,**hyper_params)
    }

    trainer_f = {
        'ConvTran' : lambda train_loader, val_loader,**kwargs : _trainer_ConvTran(train_loader,val_loader,**kwargs),
        'hydra' :lambda data, **kwargs : _trainer_hydra(data),
        'MRH' :  lambda data, **kwargs : _train_aeon(data,MultiRocketHydra(**kwargs)),
        'inceptionTime' : lambda train_data, val_data, **kwargs : _train_inceptionTime(train_data,val_data,
                                                                                       val_ratio=0.1,**kwargs)
    }

    score_f = {
        'ConvTran' : lambda model, data :model.eval().score(data),
        'hydra' : lambda model, data :(1- model.score(data).cpu().numpy().item()),
        'MRH' : lambda model, X,y : model.score(X,y),
        'inceptionTime' : lambda model, X,y:  np.sum( model.predict(X)==y ) / y.shape[0]
    }

    # use previously defined functions
    data_loader = dataloader_f[model_name](dataset,**hyper_params)

    if model_name in ['ConvTran','hydra','inceptionTime']:
        # if torch GPU model, empty cache and reset peak memory stats
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        # then train the model and get memory usage
        model = trainer_f[model_name] (*data_loader[:-1],**hyper_params)
        mem_used = {
            'peak_memory_GB': torch.cuda.max_memory_allocated()  / 1024**3,
            'average_memory_GB': torch.cuda.memory_allocated()  / 1024**3
        }
    else:
        # case for aeon classifiers
        model, mem_used = profile_function(trainer_f[model_name], data_loader[0],**hyper_params)

    accuracy = score_f[model_name]( model, data_loader[-1] ) if model_name in ['ConvTran','hydra'] \
        else score_f[model_name](model, *data_loader[-1])

    to_return = model, accuracy, mem_used

    if return_train_predictions:
        # if required, get train set predictions on a NON shuffled dataloader
        train_data = dataloader_f[model_name](dataset,only_train=True,kwargs=hyper_params)
        train_predictions = model.predict(train_data)
        to_return = (*to_return, train_predictions)

    return to_return