"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
from sklearn.cluster import KMeans

class ClusterFinder:
    def __init__(self, cluster_name='K-Means'):
        self.cluster = self._select_cluster_method(cluster_name)

    """ Input:  cluster: Clustering method instance.
                features: Array of features to generate clusters.
                save: Indicates whether to save the features as a file or not.
        Output: Array of predictions.
    
        Function that instantiates the whole model and generates the predictions."""
    def generate_clusters(self, features, save=False):
        clusters = self.cluster.fit_predict(features)

        # Save features to data.results
        if save:
            clusters.to_csv("data/results/clusters.tsv", sep='\t', index=False)

        return clusters

    """ Input:  available_device: String indicating the available device for PyTorch.
                cluster_name: String indicating selection of cluster method.
        Output: Encoder instance.

        Function that instantiates the desired encoder."""
    @staticmethod
    def _select_cluster_method(cluster_name):
        cluster_method = None

        if cluster_name == 'K-Means':
            cluster_method = KMeans()

        return cluster_method


if __name__=="__main__":
    exit(0)