"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
from . import autoEncoder

class BrainEncoder:
    def __init__(self, available_device, in_channels, encoder_name='AutoEncoder'):
        self.encoder = self._select_encoder(available_device, in_channels, encoder_name)
        self.encoder.to(available_device)

    """ Input:  dataset: Data to generate predictions.
                save: Boolean that indicates whether to save the features as a file or not.
        Output: Array of predictions.
    
        Function that instantiates the whole model and generates the predictions."""
    def generate_features(self, dataset, save=False):
        features = self.encoder.encode(dataset)
        print(features)
        print(features.shape)

        if save: features.to_csv("data/brainEncoder/features.csv", index=False)

        return features

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
        self.encoder.fit(train_loader, val_dataloader, epochs, batch_size)

    """ Input:  data_loader: Dataloader instance.
                epochs: Integer indicating the number of epochs to train the model.
                batch_size: Integer indicating the batch size.
                save: Boolean that indicates whether to save the features as a file or not.
        Output: Encoded MRI signal for the train and validation dataloaders.

        Function that trains the dataset and encodes the data. """
    def fit_transform(self, train_loader, val_dataloader, epochs, batch_size, save=False):
        self.fit(train_loader, val_dataloader, epochs, batch_size)
        return self.generate_features(train_loader, save), self.generate_features(val_dataloader, save)

if __name__=="__main__":
    exit(0)