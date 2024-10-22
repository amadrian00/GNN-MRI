"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import numpy
import torch
from . import autoEncoder

class BrainEncoder:
    def __init__(self, available_device, in_channels, encoder_name='AutoEncoder'):
        self.autoencoder = self._select_encoder(available_device, in_channels, encoder_name)
        self.autoencoder.to(available_device)
        self.available_device = available_device

    """ Input:  dataloader: Data to generate predictions.
        Output: Array of predictions.
    
        Function that instantiates the whole model and generates the predictions."""
    def transform(self, dataloader):
        with (torch.no_grad()):
            self.autoencoder.eval()
            encoded = []

            for batch in dataloader:
                batch_data, _ = batch
                encoded.append(self.autoencoder.encode(batch_data).cpu())

            encoded = numpy.concatenate(encoded, axis=0)
            return encoded

    """ Input:  available_device: String indicating the available device for PyTorch.
                encoder_name: String indicating selection of encoder layer.
                in_channels: Integer indicating the input channel for the encoder layer.
        Output: Encoder instance.
    
        Function that instantiates the desired encoder."""
    @staticmethod
    def _select_encoder(available_device,in_channels, encoder_name):
        encoder_instance = None

        if encoder_name == 'AutoEncoder':
            encoder_instance = autoEncoder.AE(available_device, in_channels)

        return encoder_instance

    """ Input:  train_loader: Instance of data to train the model.
                val_dataloader: Instance of data to validate the model.
                epochs: Integer indicating the number of epochs to train the model.
                batch_size: Integer indicating the batch size.
        Output: -.

        Function that trains the dataset. """
    def fit(self, train_loader, val_dataloader, epochs, batch_size):
        self.autoencoder.fit(train_loader, val_dataloader, epochs, batch_size)

    """ Input:  data_loader: Dataloader instance.
                epochs: Integer indicating the number of epochs to train the model.
                batch_size: Integer indicating the batch size.
                save: Boolean that indicates whether to save the features as a file or not.
        Output: Encoded MRI signal for the train and validation dataloaders.

        Function that trains the dataset and encodes the data. """
    def fit_transform(self, train_loader, val_dataloader, epochs, batch_size):
        self.fit(train_loader, val_dataloader, epochs, batch_size)
        return self.transform(train_loader), self.transform(val_dataloader)

    """ Input:  -
        Output: Loaded model.

        Function that loads the trained model. """
    def load_encoder(self, device = None):
        if device is None: device = self.available_device

        return self.autoencoder.encoder.load_state_dict(torch.load('brainEncoder/encoder.pt', weights_only=True, map_location= device))