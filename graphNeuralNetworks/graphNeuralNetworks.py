"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import graphCreation as gC

""" Input:  gnn: String indicating selection of gnn layer.
            dataset: Data to generate predictions.
    Output: GNN instance.

    Function that instantiates the desired gnn layer."""
def predict_edges(gnn, dataset):
    gnn_instance = select_gnn(gnn)
    graph = gC.create_graph(dataset)

    return gnn_instance, graph
    #return gnnInstance(graph)

""" Input:  gnn: String indicating selection of gnn layer.
    Output: GNN instance.

    Function that instantiates the desired gnn layer."""
def select_gnn(gnn = ""):
    gnn_instance = gnn
    return gnn_instance

if __name__=="__main__":
    exit(0)