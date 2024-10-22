"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import os
import torch
import argparse
from torch.utils import data
from torch.utils.data import DataLoader

from data import prepareDataset
from clustering import clusterFinder
from brainEncoder import brainEncoder
from data.prepareDataset import DallasDataSet
from sklearn.model_selection import train_test_split

def get_args():
    parser = argparse.ArgumentParser(description="A script for training a model.")

    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=10,
        help='Number of epochs for training (default: 10)'
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=32,
        help='Size of batches for training (default: 32)'
    )

    parser.add_argument(
        '-t', '--train',
        type=bool,
        default=False,
        help='Train model (default: False)'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print('Executing on:', torch.cuda.get_device_name())
        root_dir = os.path.join('/', 'DataCommon4/aayuso')

    else:
        print('Executing on: CPU')
        root_dir = 'data/datasets/'

    dataset = prepareDataset.DallasDataSet(device, root_dir=root_dir)

    column = 'CESDepression'
    count_class_0 = len(dataset.dataframe.loc[dataset.dataframe[column] == 0])
    count_class_1 = len(dataset.dataframe.loc[dataset.dataframe[column] == 1])

    # Calculate weights for each class
    weight_for_class_0 = len(dataset.dataframe) / (count_class_0 * 2)  # Majority class weight
    weight_for_class_1 = len(dataset.dataframe) / (count_class_1 * 2)  # Minority class weight

    train_df, temp_df = train_test_split(dataset.dataframe, test_size=0.2, stratify=dataset.dataframe['Alzheimer'], random_state=24)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Alzheimer'], random_state=24)

    train_df = DallasDataSet(device, df=train_df,)
    val_df = DallasDataSet(device, df=val_df)
    test_df = DallasDataSet(device, df=test_df)

    train_dataloader = data.DataLoader(train_df, batch_size=args.batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_df, batch_size=len(val_df), shuffle=True)
    test_dataloader = data.DataLoader(test_df, batch_size=len(test_df), shuffle=True)

    brainEncoder = brainEncoder.BrainEncoder(device, dataset.__getitem__(0)[0].shape,'AutoEncoder')
    save = True
    if args.train:
        brainEncoder.fit(train_dataloader, val_dataloader, args.epochs, args.batch_size)
    else:
        encoder = brainEncoder.load_encoder()

    train_encoded = brainEncoder.transform(train_dataloader)
    val_encoded = brainEncoder.transform(val_dataloader)
    test_encoded = brainEncoder.transform(test_dataloader)

    clustering = clusterFinder.ClusterFinder()
    clusters = clustering.generate_clusters(train_encoded)

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    all_dataloader = DataLoader(dataset, batch_size=args.batch_size)
    all_encoded = brainEncoder.transform(all_dataloader)
    clustering = clusterFinder.ClusterFinder()
    clusters2 = clustering.generate_clusters(all_encoded)

    X = all_encoded
    y = clusters2
    alzheimer = np.array([x for _, labels in all_dataloader for x in labels[0].numpy().tolist()])
    is_alzheimer = alzheimer == 1
    is_not_alzheimer = ~is_alzheimer

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    plt.scatter(X_pca[is_not_alzheimer, 0], X_pca[is_not_alzheimer, 1], c=y[is_not_alzheimer], cmap='viridis', s=50,
                alpha=0.7, label="Other")

    plt.scatter(X_pca[is_alzheimer, 0], X_pca[is_alzheimer, 1], marker='x', c=y[is_alzheimer], label="Alzheimer", s=100)

    plt.title("PCA projection of 32D vectors with clusters and Alzheimer Label")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    plt.colorbar(label="Cluster ID")

    plt.legend()

    plt.show()