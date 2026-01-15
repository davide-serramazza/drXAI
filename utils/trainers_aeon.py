from aeon.classification.hybrid import HIVECOTEV2
# TODO remove this!
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.interval_based import DrCIFClassifier


from models.MultiRocketHydra import MultiRocketHydra

from memory_profiler import  memory_usage



def profile_function(func, *args, **kwargs):

    # Variable to store the result
    trained_model = []

    def wrapper():
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



def train(dataset, device, batch_size, model_name, return_train_predictions=True, verbose=False):

    # TODO only some allowed clfs!!

    trainer_f = ( lambda data : _train_aeon(data,HIVECOTEV2())) if model_name == "HC2" else \
        ( lambda data : _train_aeon(data,DrCIFClassifier())) if model_name == "drCIF" else \
        ( lambda data : _train_aeon(data,MultiRocketHydra(hydra_params = {},
                                                          multiRocket_params = {})
                                    )) if model_name == "MRY" else \
        ( lambda data : _train_aeon(data,MultiRocketHydraClassifier()))   #TODO fix a

    score_f =  lambda model, X,y : model.score(X,y)

    dataloader = dataloader_aeon(dataset)
    print("training", model_name)
    model, mem_used = profile_function(trainer_f, dataloader[0])
    accuracy = score_f (model,*dataloader[1])

    to_return = model, accuracy, mem_used

    if return_train_predictions:
        train_predictions = model.predict(dataloader[0][0])
        to_return = (*to_return, train_predictions)

    return to_return



def dataloader_aeon(dataset, batch_size=-1,only_train=False):

    data_train =  dataset['train_set']['X'] , dataset['train_set']['y']

    if not only_train:
        data_test  =        dataset['test_set']['X'], dataset['test_set']['y']

    # if only_train==False, return also the test set's
    to_return = data_train if only_train else (data_train,  data_test)

    return to_return