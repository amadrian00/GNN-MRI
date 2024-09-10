"""
Adrián Ayuso Muñoz 2024-09-06 for the GNN-MRI project.
"""

""" Input:  dataset: Dataset to generate the graph.
    Output: Graph structure.

    Function that opens the datasets and creates a list of nodes."""
def create_graph(dataset):
    nodes = create_nodes(dataset)
    edges = create_edges(dataset)

    #return graph(nodes, edges)
    return nodes,edges

""" Input:  dataset: Dataset to obtain nodes.
    Output: List of nodes.

    Function that opens the datasets and creates a list of nodes."""
def create_nodes(dataset):
    nodes = []
    for element in dataset:
        nodes.index(element[0])
    return nodes


""" Input:  dataset: Dataset to obtain edges.
    Output: List of edges.

    Function that opens the datasets and creates a list of edges."""
def create_edges(dataset):
    edges = []
    for element in dataset:
        edges.index(element[0])
    return edges

if __name__=="__main__":
    exit(0)