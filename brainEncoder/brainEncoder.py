"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
from nilearn.input_data import NiftiMasker

class BrainEncoder:
    def __init__(self, encoder_name):
        self.encoder = self.select_encoder(encoder_name)
        self.features = None

    """ Input:  dataset: Data to generate predictions.
                save: Indicates whether to save the features as a file or not.
        Output: Array of predictions.
    
        Function that instantiates the whole model and generates the predictions."""
    def generate_features(self, dataset, save=False):
        self.features = self.encoder.fit_transform(dataset)
        print(self.features)
        print(self.features.shape)

        if save:
            self.features.to_csv("data/results/features.csv", index=False)

        return self.features

    """ Input:  encoder_name: String indicating selection of encoder layer.
        Output: Encoder instance.
    
        Function that instantiates the desired encoder."""
    @staticmethod
    def select_encoder(encoder_name =""):
        encoderInstance = None

        if encoder_name == 'Nifti':
            encoderInstance = NiftiMasker(smoothing_fwhm=5, memory='nilearn_cache', memory_level=1)

        return encoderInstance

if __name__=="__main__":
    exit(0)