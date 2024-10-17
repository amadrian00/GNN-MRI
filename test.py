"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import os

import nilearn.plotting
import torch
import nibabel as nib
from sympy.physics.control.control_plots import matplotlib
from torch.utils import data
from data import prepareDataset
from brainEncoder import brainEncoder
from nilearn.masking import compute_multi_epi_mask

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda':
        print('Executing on:', torch.cuda.get_device_name())
        root_dir = os.path.join('/', 'DataCommon4/aayuso')

    else:
        print('Executing on: CPU')
        root_dir = 'data/datasets/'

    batch_size = 32

    dataset = prepareDataset.DallasDataSet(device, root_dir, preprocess=True, save = True)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # nilearn.plotting.plot_roi(nib.load('data/datasets/ds004856_gen_files/mask.nii.gz'), dataset.dataframe['rfMRI'].values[90].slicer[:,:,:,153])
    # matplotlib.pyplot.show()

    print(dataset.__getitem__(0).shape)

    brainEncoder = brainEncoder.BrainEncoder(device, dataset.__getitem__(0).shape,'AutoEncoder')

    df = dataset.dataframe['rfMRI'].values

    brainEncoder.train(dataloader, batch_size)