"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class ClusterFinder:
    def __init__(self, cluster_name='K-Means'):
        self.cluster = self._select_cluster_method(cluster_name)
        self.assigned_clusters = None

    """ Input:  cluster: Clustering method instance.
                features: Array of features to generate clusters.
        Output: Array of predictions.
    
        Function that instantiates the whole model and generates the predictions."""
    def generate_clusters(self, features):
        if self.assigned_clusters is None:
            self.assigned_clusters = self.cluster.fit_predict(features)
        return self.assigned_clusters

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

    """ Input:  dataframe: DataFrame object to add clusters.
                features: Features on which to cluster.
                save: Indicates whether to save the features as a file or not.
        Output: DataFrame with a clusters column.

        Function that adds the clustering column to the DataFrame."""
    def add_clusters_to_dataframe(self, dataframe, features, save=False):
        dataframe['cluster'] = self.generate_clusters(features)

        if save:
            dataframe[['gender','label','len','sid','sub_sid','race','cluster']].to_csv('data/results/dataset.csv', index=False)

    """ Input:  x: Features of the data to plot its clusters.
                dataloader: Dataloader of the data to plot its clusters.
        Output: -

        Function that plots the clusters of the given data in 3D."""
    def plot_clusters(self, x, dataloader):
        y = self.generate_clusters(x)

        alzheimer = np.array([x for _, labels in dataloader for x in labels])
        is_alzheimer = alzheimer == 1
        is_not_alzheimer = ~is_alzheimer

        pca = PCA(n_components=3)
        x_pca = pca.fit_transform(x)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(x_pca[is_not_alzheimer, 0], x_pca[is_not_alzheimer, 1], x_pca[is_not_alzheimer, 2],
                   c=y[is_not_alzheimer], cmap='viridis', s=50, alpha=0.7, label="No disease")

        ax.scatter(x_pca[is_alzheimer, 0], x_pca[is_alzheimer, 1], x_pca[is_alzheimer, 2],
                   marker='x', c=y[is_alzheimer], label="Disease", s=100)

        ax.set_title("PCA Projection of 64D Vectors with Clusters and Disease Label")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("PCA Component 3")

        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(y)
        cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
        cbar.set_label("Cluster ID")

        ax.legend()

        plt.show()

if __name__=="__main__":
    exit(0)