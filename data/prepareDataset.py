"""
Adrián Ayuso Muñoz 2024-09-09 for the GNN-MRI project.
"""
import os
import sys
import pandas as pd
import numpy as np

# Set future behavior for downcasting
pd.set_option('future.no_silent_downcasting', True)

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

    """ Input:  dataset_path: String that indicates root path to dataset
        Output: Dataset ready to enter the pipeline.

        Function that returns the dataset with the labels."""
    @staticmethod
    def generate_dataset(dataset_path):
        dataset_and_labels = pd.read_csv(dataset_path + '/metadata.csv')
        return dataset_and_labels

class DallasDataSet(DataSet):
    def __init__(self, dataset_name):
        DataSet.__init__(self,dataset_name)

        self.root_dir = 'data/datasets/ds004856/surveys/'
        self.physical_health_path = self.root_dir + 'Template8_Physical_Health.xlsx'
        self.mental_health_path =  self.root_dir + 'Template9_Mental_Health.xlsx'
        self.psychosocial_health_path =  self.root_dir + 'Template10_Psychosocial.xlsx'
        self.participants_path = 'data/datasets/ds004856/participants.tsv'

    """ Input:  dataset_path: String that indicates root path to dataset
        Output: Dataset ready to enter the pipeline.

        Function that returns the dataset with the labels."""
    def generate_dataset(self, save=False):
        physical_health, mental_health, _ = self.excel_to_pandas()

        participants = self.load_clean_participants(self.root_dir, self.participants_path, save)
        physical_health = self.load_clean_physical(self.root_dir,physical_health, save)
        mental_health = self.load_clean_mental(self.root_dir, mental_health, save)

        data = pd.concat([participants, physical_health, mental_health], axis=1)

        w1 = data[['AgeMRI_W1', 'Sex', 'Sys1', 'Dia1', 'CESDepression1', 'Alzheimer1']]
        w2 = data[['AgeMRI_W2', 'Sex', 'Sys2', 'Dia2', 'CESDepression2', 'Alzheimer2']]
        w3 = data[['AgeMRI_W3', 'Sex', 'Sys3', 'Dia3', 'CESDepression3', 'Alzheimer3']]

        for w in [w1, w2, w3]:
            w.columns = ['Age', 'Sex', 'Sys', 'Dia', 'CESDepression', 'Alzheimer']

        dataset = pd.concat([w1, w2, w3], axis=0)

        dataset['Participant'] = dataset.index
        dataset = dataset.dropna(subset=['Age'])
        dataset = dataset.reset_index(drop=True)

        dataset['Sex'] = dataset['Sex'].replace({'f': 0, 'm': 1}).astype(int)

        dataset = dataset[['Participant', 'Age', 'Sex', 'Sys', 'Dia', 'CESDepression', 'Alzheimer']]

        if save:
            dataset.to_csv(self.root_dir + 'dataset.csv')

        return dataset

    """ Input:  save: Boolean that indicates whether to save the dataset.
        Output: Pandas datasets without redundant information.

        Function that returns the three cleaned files."""
    def excel_to_pandas(self, save=False):
        common_drop = ['ConstructName', 'ConstructNumber', 'Wave', 'HasData']

        drop = common_drop + ['NumAssess', 'Assess32', 'Assess33','Assess34', 'Assess35']
        physical_health = self.open_join_excel(self.physical_health_path, drop)

        drop = common_drop + [ 'NumTasks', 'Asses36', 'Asses37', 'Asses38']
        mental_health = self.open_join_excel(self.mental_health_path, drop)


        drop = common_drop + ['NumAssess', 'Assess39', 'Assess40', 'Assess41', 'Assess42', 'Assess42', 'Assess43',
                              'Assess44', 'Assess45', 'Assess46', 'Assess47', 'Assess48', 'Assess49', 'Assess50', 'Assess51']
        psychosocial_health = self.open_join_excel(self.psychosocial_health_path, drop)

        if save:
            mental_health.to_csv(self.root_dir + 'clean_mental_health.csv')
            physical_health.to_csv(self.root_dir + 'clean_physical_health.csv')
            psychosocial_health.to_csv(self.root_dir + 'clean_psychosocial.csv')

        return physical_health, mental_health, psychosocial_health

    """ Input:  excel_file_path: String that indicates the path to the excel file.
                drop_columns: List of the columns' names to drop.
        Output: Pandas dataframe with Excel data dropping all rows without information.

        Function that joins the pages of a given Dataset."""
    @staticmethod
    def open_join_excel(excel_file_path, drop_columns):
        excel = pd.read_excel(excel_file_path, index_col = 0, sheet_name=None)
        merged = pd.DataFrame()
        index = 1
        for sheet in excel:
            sheet = excel[sheet]
            missingData = sheet.index[sheet['HasData'] == 2].tolist()

            sheet = sheet.drop(index=missingData)
            sheet = sheet.drop(columns=drop_columns)

            sheet = sheet.add_suffix(str(index), 'columns')

            merged = pd.concat([merged, sheet], axis=1)
            index+=1
        return merged

    """ Input:  root_dir: String to the root dir of the data files.
                participants_path: String that indicates the path to the participants file.
                save: Boolean that indicates whether to save the dataset.
        Output: Pandas dataframe with participants information.

        Function prepares and returns the patients prepared Dataset."""
    @staticmethod
    def load_clean_participants(root_dir, participants_path, save=False):
        keep_columns = ['participant_id', 'AgeMRI_W1', 'AgeMRI_W2', 'AgeMRI_W3', 'Sex']

        participants_clean = pd.read_csv(participants_path, sep='\t', index_col=0, usecols=keep_columns)
        participants_clean.index = participants_clean.index.str.replace(r'sub-(\d+)', r'\1', regex=True)

        participants_clean.index = participants_clean.index.astype(int)
        participants_clean.sort_index(inplace=True)

        if save: participants_clean.to_csv(root_dir + 'participants.csv')

        return participants_clean

    """ Input:  root_dir: String to the root dir of the data files.
                physical_health: Dataset of the physical health characteristics.
                save: Boolean that indicates whether to save the dataset.
        Output: Pandas dataframe with the physical health patients' information.

        Function prepares and returns the physical health patients' prepared Dataset."""
    @staticmethod
    def load_clean_physical(root_dir, physical_health, save=False):
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

        if save: physical_health.to_csv(root_dir + 'physical_features.csv')

        return physical_health

    """ Input:  root_dir: String to the root dir of the data files.
                mental_health: Dataset of the mental health characteristics.
                save: Boolean that indicates whether to save the dataset.
        Output: Pandas dataframe with the mental health patients' information.

        Function prepares and returns the mental health patients' prepared Dataset."""
    @staticmethod
    def load_clean_mental(root_dir, mental_health, save=False):
        keep_columns = np.array(['CESDTot371', 'CESDTot372', 'CESDTot373', 'ADASTot381', 'ADASTot382', 'ADASTot383'])

        mental_health = mental_health[keep_columns]

        mental_health.loc[:, keep_columns[0:3]] = np.where(mental_health.loc[:, keep_columns[0:3]] >= 16, 1, 0)
        mental_health.loc[:, keep_columns[3:6]] = np.where(mental_health.loc[:, keep_columns[3:6]] >= 10, 1, 0)

        mental_health.columns = ['CESDepression1', 'CESDepression2', 'CESDepression3', 'Alzheimer1', 'Alzheimer2', 'Alzheimer3']

        if save: mental_health.to_csv(root_dir + 'mental_features.csv')

        return mental_health