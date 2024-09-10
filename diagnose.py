"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import brainFeatures.brainFeatures as bF
import graphNeuralNetworks.graphNeuralNetworks as gNN
import clustering.clustering as cl
import data.prepareDataset as pD

""" Input:  selected_encoder: String that indicates encoder.
            selected_cluster: String that indicates clustering method instance.
            selected_gnn: String that indicates GNN instance.
            to_predict: Data to which predictions should be generated.
    Output: Array of predictions.

    Function that instantiates the whole model and generates the predictions."""
def diagnose(selected_encoder, selected_cluster, selected_gnn, to_predict):
    encoder = bF.select_encoder(selected_encoder)
    cluster = cl.select_cluster(selected_cluster)
    gnn = gNN.select_gnn(selected_gnn)

    # diagnose and return predictions
    predictions = predict(encoder, cluster, gnn, to_predict)
    return predictions

""" Input:  encoder: Encoder instance.
            cluster: Clustering method instance.
            gnn: GNN instance.
            to_predict: Data to which predictions should be generated.
    Output: Array of predictions.

    Function that generates predictions for the data given the desired encoder, clustering method and gnn layer."""
def predict(encoder, cluster, gnn, to_predict):
    features = bF.generate_features(encoder, to_predict)
    clusters = cl.generate_clusters(cluster, features)

    to_predict = pD.add_clusters(to_predict, clusters)

    predictions = gNN.predict_edges(gnn, to_predict)
    return predictions

if __name__=="__main__":
    string_selected_encoder = ""
    string_selected_cluster = ""
    string_selected_gnn = ""
    string_selected_dataset = ""
    selected_dataset = pD.get_dataset(string_selected_dataset)

    diagnose(string_selected_encoder, string_selected_cluster, string_selected_gnn, selected_dataset)
    exit(0)