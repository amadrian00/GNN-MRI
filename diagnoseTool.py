"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import data.prepareDataset as pD
import clustering.clusterFinder as cF
import brainEncoder.brainEncoder as bE
import graphNeuralNetworks.graphNN as gNN

class DiagnoseTool:
    def __init__(self, selected_encoder, selected_cluster, selected_gnn):
        self.encoder = bE.BrainEncoder(selected_encoder)
        self.cluster = cF.ClusterFinder(selected_cluster)
        self.gnn = gNN.GraphNN(selected_gnn)

    """ Input:  to_predict: Data to which predictions should be generated.
                save: Boolean that indicates whether or not to save the intermediate results.
        Output: Array of predictions.
    
        Function that instantiates the whole model and generates the predictions."""
    def diagnose(self, to_predict, save=False):
        predictions = self.predict(self.encoder, self.cluster, self.gnn, to_predict, save)
        return predictions

    """ Input:  encoder: Encoder instance.
                cluster: Clustering method instance.
                gnn: GNN instance.
                to_predict: Data to which predictions should be generated.
                save: Boolean that indicates whether or not to save the intermediate results.
        Output: Array of predictions.
    
        Function that generates predictions for the data given the desired encoder, clustering method and gnn layer."""
    @staticmethod
    def predict(encoder, cluster, gnn, to_predict, save):
        features = encoder.generate_features(to_predict)
        clusters = cluster.generate_clusters(cluster, features)

        if save: clusters.to_csv()

        predictions = gnn.predict_edges(to_predict)
        return predictions

if __name__=="__main__":
    string_selected_encoder = ""
    string_selected_cluster = ""
    string_selected_gnn = ""
    string_selected_dataset = ""
    selected_dataset = pD.DallasDataSet("Dallas DataSet").generate_dataset()

    diagnosis = DiagnoseTool(string_selected_encoder, string_selected_cluster, string_selected_gnn)

    diagnosis.diagnose(selected_dataset)
    exit(0)