"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""

""" Input: String indicating selection of encoder layer.
    Output: Encoder instance.

    Function that instantiates the desired encoder."""
def select_encoder(encoder = ""):
    encoderInstance = encoder
    return encoderInstance

""" Input:  dataset: List of string that indicates dataset to be used.
            selected_encoder: String that indicates clustering method instance.
    Output: Array of predictions.

    Function that instantiates the whole model and generates the predictions."""
def generate_features(dataset=None, selected_encoder=""):
    if dataset is None:
        dataset = ["a"]

    features = []
    encoder = select_encoder(selected_encoder)
    return features

if __name__=="__main__":
    exit(0)