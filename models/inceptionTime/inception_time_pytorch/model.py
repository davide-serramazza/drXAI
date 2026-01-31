import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from ..inception_time_pytorch.modules import InceptionModel


class InceptionTime():
    
    def __init__(self,
                 train_data,
                 val_data,
                 filters,
                 depth,
                 models,
                 batch_size=256,
                 early_stop_counter=20):
        
        '''
        Implementation of InceptionTime model introduced in Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier,
        C., Schmidt, D.F., Weber, J., Webb, G.I., Idoumghar, L., Muller, P.A. and Petitjean, F., 2020. InceptionTime:
        Finding AlexNet for Time Series Classification. Data Mining and Knowledge Discovery, 34(6), pp.1936-1962.

        # TODO update documentation
        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        y: np.array.
            Class labels, array with shape (samples,) where samples is the number of time series.

        filters: int.
            The number of filters (or channels) of the convolutional layers of each model.

        depth: int.
            The number of blocks of each model.
        
        models: int.
            The number of models.
        '''

        # Check if GPU is available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # extract data
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        # Scale the data.
        self.mu = np.nanmean(x_train, axis=0, keepdims=True)
        self.sigma = np.nanstd(x_train, axis=0, keepdims=True)
        x_train = (x_train - self.mu) / (self.sigma + 1e-16)
        
        # Save the data.
        self.x_train = torch.from_numpy(x_train).float().to(self.device)
        self.y_train = torch.from_numpy(y_train).long().to(self.device)

        self.x_val = torch.from_numpy( (x_val - self.mu) / (self.sigma + 1e-16)  ).float().to(self.device)
        self.y_val = torch.from_numpy(y_val).long().to(self.device)

        self.batch_size = batch_size
        self.early_stop_counter = early_stop_counter

        # Build and save the models.
        self.models = [
            InceptionModel(
                input_size=x_train.shape[1],
                num_classes=len(np.unique(y_train)),
                filters=filters,
                depth=depth,
            ).to(self.device) for _ in range(models)
        ]
    
    def fit(self,
            learning_rate,
            epochs,
            verbose=True):
        
        '''
        Train the models.

        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''
        
        # Generate the training dataset.
        dataset_train = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True
        )

        dataset_val = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x_val, self.y_val),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for m in range(len(self.models)):
            
            # Define the optimizer.
            optimizer = torch.optim.Adam(self.models[m].parameters(), lr=learning_rate)
            
            # Define the loss function.
            loss_fn = torch.nn.CrossEntropyLoss()

            best_loss = np.inf ; best_accuracy = 0 ; non_improvement_count = 0

            # Train the model.
            print(f'Training model {m + 1} on {self.device}.')
            self.models[m].train(True)
            for epoch in range(epochs):
                for features, target in dataset_train:
                    optimizer.zero_grad()
                    output = self.models[m](features.to(self.device))
                    loss = loss_fn(output, target.to(self.device))
                    loss.backward()
                    optimizer.step()
                    #accuracy = (torch.argmax(torch.nn.functional.softmax(output, dim=-1), dim=-1) == target).float().sum() / target.shape[0]




                # validation step
                with torch.no_grad():
                    loss = 0.0 ; accuracy = 0.0 ; n_samples = 0
                    for features, target in dataset_val:
                        output = self.models[m](features.to(self.device))
                        loss += loss_fn(output, target.to(self.device)).item()*target.shape[0]
                        accuracy +=  torch.sum(torch.argmax(torch.nn.functional.softmax(output, dim=-1), dim=-1) == target)
                        n_samples += target.shape[0]
                    loss /= n_samples ; accuracy /= n_samples
                    if verbose:
                        print('epoch: {}, validation loss: {:,.6f},validation accuracy: {:.6f}'.format(1 + epoch, loss, accuracy))


                if loss < best_loss:
                    best_loss = loss    ;   best_accuracy = accuracy    ;     non_improvement_count = 0
                    torch.save(self.models[m], f'tmp/tmp_inceptionTime_{m + 1}.pth')
                else:
                    non_improvement_count += 1

                if non_improvement_count == self.early_stop_counter:
                    print('Early stop at epoch: {}, best loss: {:,.6f}, best accuracy: {:.6f}'.format(1 + epoch,
                                                                                                      best_loss, best_accuracy))
                    break

            self.models[m] = torch.load(f'tmp/tmp_inceptionTime_{m + 1}.pth', map_location=self.device,
                                                weights_only=False)
            self.models[m].train(False)
            print('-----------------------------------------')
    
    def predict(self, x):
        
        '''
        Predict the class labels.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        Returns:
        __________________________________
        y: np.array.
            Predicted labels, array with shape (samples,) where samples is the number of time series.
        '''

        # Scale the data.
        x = torch.from_numpy((x - self.mu) / (self.sigma + 1e-16 )).float().to(self.device)
        
        # Get the predicted probabilities.
        n_batches = ( x.shape[0]//self.batch_size ) + 1
        p = []
        with torch.no_grad():
            for i in range(0,n_batches):
                current_x = x[i*self.batch_size: min( (i+1)*self.batch_size, x.shape[0] )]
                p.append ( torch.concat([torch.nn.functional.softmax(model(current_x), dim=-1).unsqueeze(-1) \
                                  for model in self.models], dim=-1).mean(-1) )
        
        # Get the predicted labels.
        y = torch.cat(p).argmax(-1).detach().cpu().numpy().flatten()

        return y
