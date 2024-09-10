"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""

""" Input:  encoder: Encoder instance.
            dataset: Data to generate predictions.
            save: Indicates whether to save the features as a file or not.
    Output: Array of predictions.

    Function that instantiates the whole model and generates the predictions."""
def generate_features(encoder, dataset, save=False):
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