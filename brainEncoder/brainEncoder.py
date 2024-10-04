"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
from . import autoEncoder

class BrainEncoder:
    def __init__(self, in_channels, encoder_name='AutoEncoder'):
        self.encoder = self._select_encoder(encoder_name, in_channels)
        self.features = None

    """ Input:  dataset: Data to generate predictions.
                save: Boolean that indicates whether to save the features as a file or not.
        Output: Array of predictions.
    
        Function that instantiates the whole model and generates the predictions."""
    def generate_features(self, dataset, save=False):
        self.features = self.encoder.fit_transform(dataset)
        print(self.features)
        print(self.features.shape)

        if save: self.features.to_csv("data/results/features.csv", index=False)

        return self.features

    """ Input:  encoder_name: String indicating selection of encoder layer.
                in_channels: Integer indicating the input channel for the encoder layer.
        Output: Encoder instance.
    
        Function that instantiates the desired encoder."""
    @staticmethod
    def _select_encoder(encoder_name, in_channels):
        encoder_instance = None

        if encoder_name == 'AutoEncoder':
            encoder_instance = autoEncoder.AE(in_channels)

        return encoder_instance

    """ Input:  data_loader: Dataloader instance.
                batch_size: Integer indicating the batch size.
        Output: -.

        Function that trains the dataset. """
    def train(self, data_loader, batch_size):
        self.encoder.train_loop(data_loader, batch_size)


if __name__=="__main__":
    exit(0)