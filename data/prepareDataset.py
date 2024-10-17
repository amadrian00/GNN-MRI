"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import os
import re
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn.signal import clean
from nilearn.image import resample_img
from torch.utils.data import Dataset
from nilearn.masking import compute_multi_epi_mask, apply_mask

# Set future behavior for downcasting
pd.set_option('future.no_silent_downcasting', True)

class DallasDataSet(Dataset):
    def __init__(self, available_device, root_dir='data/datasets/', save=False, force_update=False, preprocess= False):
        self.mask = None
        self.mask_affine = None

        self.root_dir = root_dir + '/ds004856'
        self.available_device = available_device

        self.dataframe = self.generate_dataset(self.root_dir+'/surveys/', 'data/datasets/ds004856_gen_files/', save, force_update, preprocess)
        self.fmri_data = None
        self.clean_data = self.dataframe['clean'].values

    """ Input:  files_dir: String indicating the root directory where the dataset files are stored.
                save_dir: String indicating the directory where the generated files are stored.
                save: Boolean that indicates whether to save the dataset.
                force_update: Boolean that indicates whether to force update the dataset if it exists.
                preprocess: Boolean that indicates whether to preprocess the dataset.
        Output: PandasDataframe containing the dataset.
        
        Function that returns the dataset with the labels."""
    def generate_dataset(self, files_dir, save_dir, save, force_update, preprocess):
        if os.path.isfile(save_dir + 'dataset.csv') and not force_update: # If dataset exists and update is not forced just load it.
            dataset = pd.read_csv(save_dir + 'dataset.csv', index_col=0)

        else:
            if save and not os.path.exists(save_dir): # If the directory to save files does not exist then create it.
                os.makedirs(save_dir)

            physical_health, mental_health, _ = self._excel_to_pandas(files_dir, save_dir, save)

            participants = self._load_clean_participants(save_dir, self.root_dir + '/participants.tsv', save)
            physical_health = self._load_clean_physical(save_dir, physical_health, save)
            mental_health = self._load_clean_mental(save_dir, mental_health, save)
            fmri_paths = self._get_fmri_path(save_dir, self.root_dir, save)

            data = pd.concat([participants, physical_health, mental_health, fmri_paths], axis=1)


            w1 = data[['AgeMRI_W1', 'Sex', 'Sys1', 'Dia1', 'CESDepression1', 'Alzheimer1', 'Wave1']]
            w2 = data[['AgeMRI_W2', 'Sex', 'Sys2', 'Dia2', 'CESDepression2', 'Alzheimer2', 'Wave2']]
            w3 = data[['AgeMRI_W3', 'Sex', 'Sys3', 'Dia3', 'CESDepression3', 'Alzheimer3', 'Wave3']]

            for w in [w1, w2, w3]: # Create one dataset per wave.
                w.columns = ['Age', 'Sex', 'Sys', 'Dia', 'CESDepression', 'Alzheimer','rfMRI']

            dataset = pd.concat([w1, w2, w3], axis=0) # Join the wave datasets as one.

            dataset['Participant'] = dataset.index
            dataset = dataset.dropna(subset=['Age'])
            dataset = dataset.dropna(subset=['rfMRI'])
            dataset = dataset.reset_index(drop=True)

            dataset['Sex'] = dataset['Sex'].replace({'f': 0, 'm': 1}).astype(int)

            dataset = dataset[['Participant', 'Age', 'Sex', 'Sys', 'Dia', 'CESDepression', 'Alzheimer', 'rfMRI']]

        dataset['file_rfMRI'] = np.array(list(map(lambda x: nib.load(x), dataset['rfMRI'].values))) # Load image files.
        self.fmri_data = dataset['file_rfMRI'].values

        if  not os.path.isfile(save_dir + 'mask.nii.gz') or force_update: # If mask does not exist or force update then create it.
            self.mask_affine = dataset['file_rfMRI'].values[0].affine
            self.mask = compute_multi_epi_mask(dataset['file_rfMRI'].values, n_jobs=-1, target_shape=(64, 64, 43),
                                          target_affine=dataset['file_rfMRI'].values[0].affine, threshold=0.22)
            np.save(save_dir + 'mask_affine.npy', self.mask_affine)
            nib.save(self.mask, save_dir + 'mask.nii.gz')
        else:
            self.mask = nib.load(save_dir + 'mask.nii.gz')
            self.mask_affine = np.load(save_dir + 'mask_affine.npy')

        if preprocess or not os.path.isfile(save_dir + 'dataset.csv') or force_update: # If explicitly requested preprocess or the dataset file does not exist or force update, then preprocess the dataset.
            self._preprocess(dataset, self.fmri_data, self.mask_affine, self.mask)

        if save:
            cols_to_save = ['Participant', 'Age', 'Sex', 'Sys', 'Dia', 'CESDepression', 'Alzheimer', 'rfMRI', 'clean']
            dataset[cols_to_save].to_csv(save_dir + 'dataset.csv')

        dataset['clean'] = np.array(list(map(lambda x: np.load(x, mmap_mode='r'), dataset['clean'].values)))

        print('Dallas dataset has been generated correctly.')
        return dataset

    """ Input:  files_dir: String indicating the root directory where the dataset files are stored.
                save_dir: String indicating the directory where the generated files are stored.
                save: Boolean that indicates whether to save the dataset.
        Output: Pandas datasets without redundant information.

        Function that returns the three cleaned files."""
    def _excel_to_pandas(self,files_dir, save_dir, save):
        common_drop = ['ConstructName', 'ConstructNumber', 'Wave', 'HasData']

        drop = common_drop + ['NumAssess', 'Assess32', 'Assess33','Assess34', 'Assess35']
        physical_health = self._open_join_excel(files_dir + 'Template8_Physical_Health.xlsx', drop)

        drop = common_drop + [ 'NumTasks', 'Asses36', 'Asses37', 'Asses38']
        mental_health = self._open_join_excel(files_dir + 'Template9_Mental_Health.xlsx', drop)


        drop = common_drop + ['NumAssess', 'Assess39', 'Assess40', 'Assess41', 'Assess42', 'Assess42', 'Assess43',
                              'Assess44', 'Assess45', 'Assess46', 'Assess47', 'Assess48', 'Assess49', 'Assess50', 'Assess51']
        psychosocial_health = self._open_join_excel(files_dir + 'Template10_Psychosocial.xlsx', drop)

        if save:
            mental_health.to_csv(save_dir + 'clean_mental_health.csv')
            physical_health.to_csv(save_dir + 'clean_physical_health.csv')
            psychosocial_health.to_csv(save_dir + 'clean_psychosocial.csv')

        return physical_health, mental_health, psychosocial_health

    """ Input:  excel_file_path: String that indicates the path to the excel file.
                drop_columns: List of the columns' names to drop.
        Output: Pandas dataframe with Excel data dropping all rows without information.

        Function that joins the pages of a given Dataset."""
    @staticmethod
    def _open_join_excel(excel_file_path, drop_columns):
        try:
            excel = pd.read_excel(excel_file_path, index_col=0, sheet_name=None)
        except Exception as e:
            raise ValueError(f"Error reading the Excel file: {e}")

        merged = pd.DataFrame()
        for index, (sheet_name, data_frame) in enumerate(excel.items(), start=1):
            missing_data = data_frame[data_frame['HasData'] == 2].index

            data_frame = data_frame.drop(index=missing_data).drop(columns=drop_columns, errors='ignore')

            data_frame = data_frame.add_suffix(f'{index}')

            merged = pd.concat([merged, data_frame], axis=1)

        return merged

    """ Input:  save_dir: String indicating the directory where the generated files are stored.
                participants_path: String that indicates the path to the participants file.
                save: Boolean that indicates whether to save the dataset.
        Output: Pandas dataframe with participants information.

        Function prepares and returns the patients prepared Dataset."""
    @staticmethod
    def _load_clean_participants(save_dir, participants_path, save=False):
        keep_columns = ['participant_id', 'AgeMRI_W1', 'AgeMRI_W2', 'AgeMRI_W3', 'Sex']

        participants_clean = pd.read_csv(participants_path, sep='\t', index_col=0, usecols=keep_columns)
        participants_clean.index = participants_clean.index.str.replace(r'sub-(\d+)', r'\1', regex=True)

        participants_clean.index = participants_clean.index.astype(int)
        participants_clean.sort_index(inplace=True)

        if save: participants_clean.to_csv(save_dir + 'participants.csv')

        return participants_clean

    """ Input:  save_dir: String indicating the directory where the generated files are stored.
                physical_health: Dataset of the physical health characteristics.
                save: Boolean that indicates whether to save the dataset.
        Output: Pandas dataframe with the physical health patients' information.

        Function prepares and returns the physical health patients' prepared Dataset."""
    @staticmethod
    def _load_clean_physical(save_dir, physical_health, save=False):
        keep_columns = np.array(['BPDay1Time1Sys341', 'BPDay1Time1Sys342', 'BPDay1Time1Sys343',
                   'BPDay1Time1Dia341', 'BPDay1Time1Dia342', 'BPDay1Time1Dia343',
                   'BPDay1Time2Sys341', 'BPDay1Time2Sys342', 'BPDay1Time2Sys343',
                   'BPDay1Time2Dia341', 'BPDay1Time2Dia342', 'BPDay1Time2Dia343',

                   'BPDay2Time1Sys341', 'BPDay2Time1Sys342', 'BPDay2Time1Sys343',
                   'BPDay2Time1Dia341', 'BPDay2Time1Dia342', 'BPDay2Time1Dia343',
                   'BPDay2Time2Sys341', 'BPDay2Time2Sys342', 'BPDay2Time2Sys343',
                   'BPDay2Time2Dia341', 'BPDay2Time2Dia342', 'BPDay2Time2Dia343'])

        physical_health = physical_health[keep_columns]

        sys1 = physical_health[keep_columns[[0,6,12,18]]].mean(axis=1)
        sys2 = physical_health[keep_columns[[1,7,13,19]]].mean(axis=1)
        sys3 = physical_health[keep_columns[[2,8,14,20]]].mean(axis=1)

        dia1 = physical_health[keep_columns[[3,9,15,21]]].mean(axis=1)
        dia2 = physical_health[keep_columns[[4,10,16,22]]].mean(axis=1)
        dia3 = physical_health[keep_columns[[5,11,17,23]]].mean(axis=1)

        physical_health =  pd.concat([sys1, dia1, sys2, dia2, sys3, dia3 ], axis=1)

        physical_health.columns = ['Sys1', 'Dia1', 'Sys2', 'Dia2', 'Sys3', 'Dia3']

        if save: physical_health.to_csv(save_dir + 'physical_features.csv')

        return physical_health

    """ Input:  save_dir: String indicating the directory where the generated files are stored.
                mental_health: Dataset of the mental health characteristics.
                save: Boolean that indicates whether to save the dataset.
        Output: Pandas dataframe with the mental health patients' information.

        Function prepares and returns the mental health patients' prepared Dataset."""
    @staticmethod
    def _load_clean_mental(save_dir, mental_health, save=False):
        keep_columns = np.array(['CESDTot371', 'CESDTot372', 'CESDTot373', 'ADASTot381', 'ADASTot382', 'ADASTot383'])

        mental_health = mental_health[keep_columns]

        mental_health.loc[:, keep_columns[0:3]] = np.where(mental_health.loc[:, keep_columns[0:3]] >= 16, 1, 0)
        mental_health.loc[:, keep_columns[3:6]] = np.where(mental_health.loc[:, keep_columns[3:6]] >= 10, 1, 0)

        mental_health.columns = ['CESDepression1', 'CESDepression2', 'CESDepression3', 'Alzheimer1', 'Alzheimer2', 'Alzheimer3']

        if save: mental_health.to_csv(save_dir + 'mental_features.csv')

        return mental_health

    """ Input:  save_dir: String indicating the directory where the generated files are stored.
                root_dir: String indicating the root directory where the dataset files are stored.
                save: Boolean that indicates whether to save the dataset.
        Output: Array containing the Strings of the fMRI paths.
    
        Function that returns the path to the fMRI files."""
    @staticmethod
    def _get_fmri_path(save_dir, root_dir, save=False):
        paths = pd.DataFrame(columns=['Patient', 'Wave1', 'Wave2', 'Wave3'])

        for subdir in Path(root_dir).glob('sub-*/*/func'):
            file_dir = str(subdir.parent)
            wave = re.search(r'ses-wave(\d+)', file_dir).group(1)
            patient = re.search(r'sub-(\d+)', file_dir).group(1)

            file_path = list(subdir.glob('*-rest_run-*_bold.nii.gz'))
            file = None
            if len(file_path) > 0:
                file = f"{file_dir}/func/{file_path[-1].name}" #If 2 runs for the same wave just get the last.

            if paths['Patient'].eq(patient).any():
                paths.loc[paths['Patient'] == patient, f'Wave{wave}'] = file
            else:
                paths.loc[len(paths), ['Patient','Wave'+wave]] = [patient, file]

        paths.set_index('Patient', inplace=True)
        paths.index = paths.index.astype(int)
        paths.index.name = 'S#'
        paths.sort_index(inplace=True)

        if save: paths.to_csv(f"{save_dir}fmri_paths.csv")

        return paths

    """ Input:  dataset: Dataset instance.
                clusters: Assigned clusters.
        Output: Dataset instance.

        Function that appends the assigned cluster to each element."""
    @staticmethod
    def add_clusters(dataset, clusters):
        dataset.append(clusters)
        return dataset

    """ Input:  dataframe: Dataframe instance.
                fmri_data: FMRI image object's array.
                mask_affine: Mask affine transformation matrix.
                mask: Mask image.
        Output: -

        Function that preprocesses the fMRI files or adds their paths to the dataframe."""
    @staticmethod
    def _preprocess(dataframe, fmri_data, mask_affine, mask):
        for index, elem in dataframe.iterrows():
            path_dir = re.match(r"(.*)/", elem['rfMRI']).group(0)
            smooth_fmri = resample_img(fmri_data[index].slicer[:, :, :, :124:2], target_shape=(64, 64, 43),
                                 target_affine=mask_affine)
            clean_data = clean(apply_mask(smooth_fmri, mask, smoothing_fwhm=6), standardize='zscore_sample')
            np.save(path_dir+'clean_signal.npy', clean_data.reshape(-1).astype('float32'))

            dataframe.at[index, 'clean'] = path_dir+'clean_signal.npy'

    """ Input:  idx: Integer indicating the index of the item to get.
        Output: Time of the dataset.

        Function that returns the item at the given index of the dataset."""
    def __getitem__(self, idx):
        return torch.tensor(self.clean_data[idx], dtype=torch.float32, device=self.available_device)

    """ Input:  
        Output: Length of the dataset.

        Function that returns the length of the dataset."""
    def __len__(self):
        return len(self.fmri_data)