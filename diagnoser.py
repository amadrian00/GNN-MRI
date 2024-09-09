"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""


""" Input: String indicating selection of encoder layer.
    Output: Encoder instance.

    Function that instantiates the desired encoder."""
def select_encoder(encoder = ""):
    encoderInstance = encoder
    return encoderInstance

""" Input: String indicating selection of clustering method.
    Output: Clustering method instance.

    Function that instantiates the desired clustering method."""
def select_cluster(cluster = ""):
    clusterInstance = cluster
    return clusterInstance

""" Input: String indicating selection of gnn layer.
    Output: GNN instance.

    Function that instantiates the desired gnn layer."""
def select_gnn(gnn = ""):
    gnnInstance = gnn
    return gnnInstance

""" Input: Encoder, cluster and gnn instances.
    Output: Array of predictions.

    Function that generates predictions for the data given the desired encoder, clustering method and gnn layer."""
def diagnose(encoder, cluster, gnn):
    predictions = []
    return predictions

""" Input: String indicating selection of encoder, clustering method and gnn layer.
    Output: Array of predictions.

    Function that instantiates the whole model and generates the predictions."""
def diagnoser(selected_encoder, selected_cluster, selected_gnn):
    encoder = select_encoder(selected_encoder)
    cluster = select_cluster(selected_cluster)
    gnn = select_gnn(selected_gnn)

    # diagnose and return predictions
    predictions = diagnose(encoder, cluster, gnn)

if __name__=="__main__":
    string_selected_encoder = ""
    string_selected_cluster = ""
    string_selected_gnn = ""

    diagnoser(string_selected_encoder, string_selected_cluster, string_selected_gnn)
    exit(0)