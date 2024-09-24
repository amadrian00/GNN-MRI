"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import os
import sys
import pandas as pd

class DataSet:
    def __init__(self,dataset_name):
        self.dataset_name = dataset_name
        self.dataset_path = './data/datasets/' + self.dataset_name
        self.dataset = None

    """ Input:  -
        Output: Dataset instance.
    
        Function that prepares and returns the dataset."""
    def get_dataset(self):

        if not os.path.isdir(self.dataset_path):
            print(f" {'\033[31m'} Error: The dataset '{self.dataset_path}' does not exist.")
            sys.exit(1)  # Exit the script with a non-zero status indicating an error

        self.dataset = self.generate_dataset(self.dataset_path)
        return self.dataset

    """ Input:  dataset: Dataset instance.
                clusters: Assigned clusters.
        Output: Dataset instance.
    
        Function that appends the assigned cluster to each element."""
    @staticmethod
    def add_clusters(dataset, clusters):
        dataset.append(clusters)
        return dataset


class DallasDataSet(DataSet):
    def __init__(self, dataset_name):
        DataSet.__init__(self,dataset_name)

    """ Input:  dataset_path: String that indicates root path to dataset
        Output: Dataset ready to enter the pipeline.

        Function that returns the dataset with the labels."""

    @staticmethod
    def generate_dataset(dataset_path):
        dataset_and_labels = pd.read_csv(dataset_path + '/metadata.csv', usecols=["Subject", "Sex", "Age"])
        return dataset_and_labels



if __name__=="__main__":
    exit(0)