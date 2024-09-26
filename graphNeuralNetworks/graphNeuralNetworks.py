"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import graphCreation as gC

class GraphNN:
    def __init__(self, gnn):
        self.gnn = gnn
    """ Input:  gnn: String indicating selection of gnn layer.
                dataset: Data to generate predictions.
        Output: GNN instance.
    
        Function that instantiates the desired gnn layer."""
    def predict_edges(self, dataset):
        graph = gC.create_graph(dataset)

        return self.gnn, graph
        #return gnnInstance(graph)


if __name__=="__main__":
    exit(0)