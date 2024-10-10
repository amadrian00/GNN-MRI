"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import os
import torch
from torch.utils import data
from data import prepareDataset
from brainEncoder import brainEncoder

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Executing on:', torch.cuda.get_device_name())

    batch_size = 16
    path = os.path.abspath('/DataCommon')
    print(os.listdir(path))

    """
    dataset = prepareDataset.DallasDataSet(device)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    brainEncoder = brainEncoder.BrainEncoder(device, dataset.__getitem__(0),'AutoEncoder')

    brainEncoder.train(dataloader, batch_size)"""