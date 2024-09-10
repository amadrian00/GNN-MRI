"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import data.getDataset as gD

""" Input:  encoder: Encoder instance.
            dataset_name: List of string that indicates dataset to be used.
            save: Indicates whether to save the features as a file or not.
    Output: Array of predictions.

    Function that instantiates the whole model and generates the predictions."""
def generate_features(encoder, dataset_name=None, save=False):
    if dataset_name is None:
        dataset_name = ["a"]

    dataset = gD.get_dataset(dataset_name)
    features = encoder(dataset)

    # Save features to data.results
    if save:
        features.to_csv("data/results/features.csv", index=False)

    return features

""" Input:  encoder: String indicating selection of encoder layer.
    Output: Encoder instance.

    Function that instantiates the desired encoder."""
def select_encoder(encoder = ""):
    encoderInstance = encoder
    return encoderInstance

if __name__=="__main__":
    exit(0)