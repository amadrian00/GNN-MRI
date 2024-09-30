"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
from . import autoEncoder

class BrainEncoder:
    def __init__(self, dataset, encoder_name='AutoEncoder'):
        self.encoder = self._select_encoder(encoder_name, dataset[0].shape)
        self.features = None
        self.dataset = dataset

    """ Input:  dataset: Data to generate predictions.
                save: Indicates whether to save the features as a file or not.
        Output: Array of predictions.
    
        Function that instantiates the whole model and generates the predictions."""
    def generate_features(self, dataset, save=False):
        self.features = self.encoder.fit_transform(dataset)
        print(self.features)
        print(self.features.shape)

        if save: self.features.to_csv("data/results/features.csv", index=False)

        return self.features

    """ Input:  dims: Integer indicating the dimensions of the data.
                encoder_name: String indicating selection of encoder layer.
        Output: Encoder instance.
    
        Function that instantiates the desired encoder."""
    @staticmethod
    def _select_encoder(dims, encoder_name):
        encoder_instance = None

        if encoder_name == 'AutoEncoder':
            encoder_instance = autoEncoder.AE(dims)

        return encoder_instance

    def train(self):
        self.encoder.training(self.dataset)


if __name__=="__main__":
    exit(0)