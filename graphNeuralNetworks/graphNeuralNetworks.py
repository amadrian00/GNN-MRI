"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import graphCreation as gC

""" Input:  gnn: String indicating selection of gnn layer.
            dataset: Data to generate predictions.
    Output: GNN instance.

    Function that instantiates the desired gnn layer."""
def predict_edges(gnn, dataset):
    gnnInstance = select_gnn(gnn)
    graph = gC.create_graph(dataset)

    return gnnInstance, graph
    #return gnnInstance(graph)

""" Input:  gnn: String indicating selection of gnn layer.
    Output: GNN instance.

    Function that instantiates the desired gnn layer."""
def select_gnn(gnn = ""):
    gnnInstance = gnn
    return gnnInstance

if __name__=="__main__":
    exit(0)