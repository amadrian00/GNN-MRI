"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import data.prepareDataset as pD
import clustering.clusterFinder as cF
import brainEncoder.brainEncoder as bE
import graphNeuralNetworks.graphNN as gNN

class DiagnoseTool:
    def __init__(self, selected_encoder, selected_cluster, selected_gnn, dataset_path):
        in_channels = 0
        self.encoder = bE.BrainEncoder(in_channels, selected_encoder)
        self.cluster = cF.ClusterFinder(selected_cluster)
        self.gnn = gNN.GraphNN(selected_gnn)

    """ Input:  to_predict: Data to which predictions should be generated.
                save: Boolean that indicates whether or not to save the intermediate results.
        Output: Array of predictions.
    
        Function that generates predictions for the data given the dataset."""
    def predict(self, to_predict, save=False):
        features = self.encoder.generate_features(to_predict)
        clusters = self.cluster.generate_clusters(features)

        if save: clusters.to_csv()

        predictions = self.gnn.predict_edges(to_predict)
        return predictions

if __name__=="__main__":
    string_selected_encoder = ""
    string_selected_cluster = ""
    string_selected_gnn = ""
    string_selected_dataset = ""

    dataset = pD.DallasDataSet("Dallas DataSet")
    selected_dataset_paths = dataset.generate_dataset()['rfMRI'].values
    print(selected_dataset_paths)

    diagnosis = DiagnoseTool(string_selected_encoder, string_selected_cluster, string_selected_gnn, selected_dataset_paths)
    diagnosis.predict(selected_dataset_paths)

    exit(0)