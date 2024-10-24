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
from data.prepareDataset import DallasDataSet, Oasis3
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

def prepare_dallas_dataset():
    dallas_dataset = prepareDataset.DallasDataSet(device, root_dir=root_dir_dallas)

    column = 'CESDepression'
    count_class_0 = len(dallas_dataset.dataframe.loc[dallas_dataset.dataframe[column] == 0])
    count_class_1 = len(dallas_dataset.dataframe.loc[dallas_dataset.dataframe[column] == 1])

    # Calculate weights for each class
    weight_for_class_0 = len(dallas_dataset.dataframe) / (count_class_0 * 2)  # Majority class weight
    weight_for_class_1 = len(dallas_dataset.dataframe) / (count_class_1 * 2)  # Minority class weight

    train_dataframe, temp_dataframe = train_test_split(dallas_dataset.dataframe, test_size=0.2, stratify=dallas_dataset.dataframe['Alzheimer'],
                                                random_state=24)
    val_dataframe, test_dataframe = train_test_split(temp_dataframe, test_size=0.5, stratify=temp_dataframe['Alzheimer'], random_state=24)

    train_dataframe = DallasDataSet(device, df=train_dataframe,)
    val_dataframe = DallasDataSet(device, df=val_dataframe)
    test_dataframe = DallasDataSet(device, df=test_dataframe)

    return dallas_dataset, train_dataframe, val_dataframe, test_dataframe

def prepare_oasis_dataset():
    oasis_dataset = prepareDataset.Oasis3(device)

    train_dataframe, temp_dataframe = train_test_split(oasis_dataset.dataframe, test_size=0.2, stratify=oasis_dataset.dataframe['label'],
                                                random_state=24)
    val_dataframe, test_dataframe = train_test_split(temp_dataframe, test_size=0.5, stratify=temp_dataframe['label'], random_state=24)

    train_dataframe = Oasis3(device, df=train_dataframe, )
    val_dataframe = Oasis3(device, df=val_dataframe)
    test_dataframe = Oasis3(device, df=test_dataframe)

    return oasis_dataset, train_dataframe, val_dataframe, test_dataframe

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print('Executing on:', torch.cuda.get_device_name())
        root_dir_dallas = os.path.join('/', 'DataCommon4/aayuso')
        root_dir_oasis = '/DataCommon4/fMRI_data/OASIS3/'

    else:
        print('Executing on: CPU')
        root_dir_dallas = 'data/datasets/'
        root_dir_oasis = ''

    # dataset, train_df, val_df, test_df = prepare_dallas_dataset()
    dataset, train_df, val_df, test_df = prepare_oasis_dataset()

    train_dataloader = data.DataLoader(train_df, batch_size=args.batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_df, batch_size=len(val_df), shuffle=True)
    test_dataloader = data.DataLoader(test_df, batch_size=len(test_df), shuffle=True)

    brainEncoder = brainEncoder.BrainEncoder(device, dataset.__getitem__(0)[0].shape,'AutoEncoder')
    save = True
    if args.train:
        brainEncoder.fit(train_dataloader, val_dataloader, args.epochs, args.batch_size)
    else:
        encoder = brainEncoder.load_encoder()

    clustering = clusterFinder.ClusterFinder()
    all_dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # clustering.plot_clusters(brainEncoder.transform(all_dataloader), all_dataloader)
    clustering.add_clusters_to_dataframe(dataset.dataframe, brainEncoder.transform(all_dataloader))
