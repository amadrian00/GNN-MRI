"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import os
import sys
import pandas as pd

""" Input:  dataset_name: List of string that indicates dataset to be used.
    Output: Dataset instance.

    Function that prepares and returns the dataset."""
def get_dataset(dataset_name):
    dataset_path = './data/datasets/' + dataset_name
    if not os.path.isdir(dataset_path):
        print(f" {'\033[31m'} Error: The dataset '{dataset_path}' does not exist.")
        sys.exit(1)  # Exit the script with a non-zero status indicating an error

    dataset = generate_dataset(dataset_path)
    return dataset

""" Input:  dataset: Dataset instance.
            clusters: Assigned clusters.
    Output: Dataset instance.

    Function that appends the assigned cluster to each element."""
def add_clusters(dataset, clusters):
    dataset.append(clusters)
    return dataset

""" Input:  dataset_path: Path to the dataset directory.
    Output: Dataset ready to enter the pipeline.

    Function that returns the dataset."""
def generate_dataset(dataset_path):
    metadata = pd.read_csv(dataset_path+'/metadata.csv', usecols=["Subject", "Sex","Age"])

    return metadata

if __name__=="__main__":
    exit(0)