from models.MultiRocketHydra import MultiRocketHydra

from models.aaltd2024.code.hydra_gpu import HydraMultivariateGPU
from models.aaltd2024.code.ridge import RidgeClassifier
from models.aaltd2024.code.utils import *

from models.convTran import train_ConvTran, build_ConvTran_model, default_hyperparams as ConvTran_default_hyperparams

from models.inceptionTime.inception_time_pytorch.model import InceptionTime

from utils.data_utils import load_data_ConvTran, dataloader_hydra, dataloader_aeon

from memory_profiler import  memory_usage

exceptions = {
    ('MRH' 'AudioMNIST')  : {  'hydra_params' : {'n_kernels' : 2,'n_groups' : 32}, 'multiRocket_params' : {'n_kernels' : 781}},
    ('MRH', 'MosquitoSound')  : {  'hydra_params' : {'n_kernels' : 2}, 'multiRocket_params' : {'n_kernels' : 1532}},
    ('MRH', 'PAMAP2')  :  { 'sklearn_classifier' : True} ,
    ('ConvTran' , 'CornellWhaleChallenge')  : {  'batch_size' : 8},
    ('ConvTran' , 'FruitFlies')  : {  'batch_size' : 6},
    ('ConvTran' , 'MosquitoSound') : {  'batch_size' : 12},
    ('hydra', 'AudioMNIST')  :  { 'batch_size' : 64} ,
    ('inceptionTime', 'FruitFlies')  :  { 'batch_size' : 128} ,
    ('inceptionTime', 'AudioMNIST')  :  { 'batch_size' : 16}
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
    batch_size = 256 if 'batch_size' not in kwargs else kwargs['batch_size']
    model = InceptionTime(train_data, val_data,filters=32, depth=6, models=5)
    model.fit(learning_rate=1e-3,batch_size=batch_size,epochs=100)

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
    key = (model_name,dataset['name'])
    hyper_params = exceptions[key] if key in exceptions else {}

    # set data loaders, trainer and score functions according to current model

    dataloader_f = load_data_ConvTran if model_name=='ConvTran' else \
        dataloader_hydra if model_name=="hydra" else \
        (lambda data: dataloader_aeon(data,val_ratio=0.1,**hyper_params)) if model_name=='inceptionTime' else \
        dataloader_aeon

    trainer_f = (lambda data: _trainer_hydra(data)) if model_name=='hydra' else \
        (lambda train_loader, val_loader : _trainer_ConvTran(train_loader,val_loader,**hyper_params)) if model_name=='ConvTran' else \
        (lambda train_data, val_data : _train_inceptionTime(train_data,val_data,**hyper_params)) if model_name=='inceptionTime' else \
        ( lambda data : _train_aeon(data,MultiRocketHydra(**hyper_params)) )       # TODO each possible aeon classifiers


    score_f = (lambda model, data :(1- model.score(data).cpu().numpy().item()) ) if model_name=='hydra' else \
        (lambda model, data: model.eval().score(data)) if model_name=='ConvTran' else \
        (lambda model, X,y:  np.sum( model.predict(X)==y ) / y.shape[0] ) if model_name=='inceptionTime' else \
        (lambda model, X,y : model.score(X,y) )     #aeon classifiers case

    # use previously defined functions
    data_loader = dataloader_f(dataset)

    if model_name in ['ConvTran','hydra','inceptionTime']:
        # if torch GPU model, empty cache and reset peak memory stats
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        # then train the model and get memory usage
        model = trainer_f(*data_loader[:-1])
        mem_used = {
            'peak_memory_GB': torch.cuda.max_memory_allocated()  / 1024**3,
            'average_memory_GB': torch.cuda.memory_allocated()  / 1024**3
        }
    else:
        # case for aeon classifiers
        model, mem_used = profile_function(trainer_f, data_loader[0])

    # TODO can it be more clean??
    accuracy = score_f( model, data_loader[1] ) if model_name in ['ConvTran','hydra'] else score_f(model,*data_loader[1])

    to_return = model, accuracy, mem_used

    if return_train_predictions:
        # if required, get train set predictions on a NON shuffled dataloader
        train_data = dataloader_f(dataset,only_train=True,kwargs=hyper_params)
        train_predictions = model.predict(train_data)
        to_return = (*to_return, train_predictions)

    return to_return