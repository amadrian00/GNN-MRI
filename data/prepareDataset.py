"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""

""" Input:  dataset_name: List of string that indicates dataset to be used.
    Output: Dataset instance.

    Function that prepares and returns the dataset."""
def get_dataset(dataset_name):
    dataset_path = get_dataset_path(dataset_name)
    dataset = open(dataset_path)
    return dataset

""" Input:  dataset: Dataset instance.
            clusters: Assigned clusters.
    Output: Dataset instance.

    Function that appends the assigned cluster to each element."""
def add_clusters(dataset, clusters):
    dataset.append(clusters)
    return dataset

""" Input:  dataset_name: List of string that indicates dataset to be used.
    Output: Dataset path.

    Function that returns the dataset path."""
def get_dataset_path(dataset_name):
    if dataset_name == "a":
        dataset_path = "./datasets/A/"
    else:
        dataset_path = "./datasets/B/"
    return dataset_path

if __name__=="__main__":
    exit(0)