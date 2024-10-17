"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import os
import torch
import argparse
from torch.utils import data
from data import prepareDataset
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

    train_df, temp_df = train_test_split(dataset.dataframe, test_size=0.2, stratify=dataset.dataframe['Alzheimer'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Alzheimer'], random_state=42)

    train_df = DallasDataSet(device, df=train_df,)
    val_df = DallasDataSet(device, df=val_df)
    test_df = DallasDataSet(device, df=test_df)

    train_dataloader = data.DataLoader(train_df, batch_size=args.batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_df, batch_size=args.batch_size, shuffle=True)
    test_dataloader = data.DataLoader(test_df, batch_size=args.batch_size, shuffle=True)

    brainEncoder = brainEncoder.BrainEncoder(device, dataset.__getitem__(0)[0].shape,'AutoEncoder')

    brainEncoder.train(train_dataloader, val_dataloader, args.epochs, args.batch_size)