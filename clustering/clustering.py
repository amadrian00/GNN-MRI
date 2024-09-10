"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""

""" Input:  cluster: Clustering method instance.
            features: Array of features.
            save: Indicates whether to save the features as a file or not.
    Output: Array of predictions.

    Function that instantiates the whole model and generates the predictions."""
def generate_clusters(cluster, features, save=False):
    clusters = cluster(features)

    # Save features to data.results
    if save:
        clusters.to_csv("data/results/clusters.tsv", sep='\t', index=False)

    return clusters

""" Input:  cluster:String indicating selection of clustering method.
    Output: Clustering method instance.

    Function that instantiates the desired clustering method."""
def select_cluster(cluster = ""):
    clusterInstance = cluster
    return clusterInstance

if __name__=="__main__":
    exit(0)