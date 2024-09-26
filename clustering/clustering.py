"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
class Cluster:
    def __init__(self, cluster):
        self.cluster = cluster

    """ Input:  cluster: Clustering method instance.
                features: Array of features to generate clusters.
                save: Indicates whether to save the features as a file or not.
        Output: Array of predictions.
    
        Function that instantiates the whole model and generates the predictions."""
    def generate_clusters(self, features, save=False):
        clusters = self.cluster(features)

        # Save features to data.results
        if save:
            clusters.to_csv("data/results/clusters.tsv", sep='\t', index=False)

        return clusters

if __name__=="__main__":
    exit(0)