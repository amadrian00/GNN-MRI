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

    """ Input:  dataset_path: String that indicates root path to dataset
        Output: Dataset ready to enter the pipeline.

        Function that returns the dataset with the labels."""
    @staticmethod
    def generate_dataset(dataset_path):
        dataset_and_labels = pd.read_csv(dataset_path + '/metadata.csv', usecols=["Subject", "Sex", "Age"])
        return dataset_and_labels

    def old_to_clean(self):
        drop = ['ConstructName','ConstructNumber','Wave','HasData','NumAssess', 'Assess32', 'Assess33','Assess34', 'Assess35']
        physical_health = self.open_join_excel('data/datasets/ds004856/surveys/Template8_Physical_Health.xlsx', drop)
        physical_health.to_csv('data/datasets/ds004856/surveys/clean_physical_health.csv')

        drop = ['ConstructName', 'ConstructNumber', 'Wave', 'HasData', 'NumTasks', 'Asses36', 'Asses37', 'Asses38']
        mental_health = self.open_join_excel('data/datasets/ds004856/surveys/Template9_Mental_Health.xlsx', drop)
        mental_health.to_csv('data/datasets/ds004856/surveys/clean_mental_health.csv')

        drop = ['ConstructName', 'ConstructNumber', 'Wave', 'HasData', 'NumAssess', 'Assess39', 'Assess40', 'Assess41',
                'Assess42', 'Assess42', 'Assess43', 'Assess44', 'Assess45', 'Assess46', 'Assess47', 'Assess48', 'Assess49'
                , 'Assess50', 'Assess51']
        psychosocial_health = self.open_join_excel('data/datasets/ds004856/surveys/Template10_Psychosocial.xlsx', drop)
        psychosocial_health.to_csv('data/datasets/ds004856/surveys/clean_psychosocial.csv')

        return physical_health, mental_health, psychosocial_health

    def open_join_excel(self, excel_file, drop_columns):
        excel = pd.read_excel(excel_file, index_col = 0, sheet_name=None)
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

    def load_clean_physical(self):
        keep_columns = ['S#',
                   'BPDay1Time1Sys341', 'BPDay1Time1Sys342', 'BPDay1Time1Sys343',
                   'BPDay1Time1Dia341', 'BPDay1Time1Dia342', 'BPDay1Time1Dia343',
                   'BPDay1Time2Sys341', 'BPDay1Time2Sys342', 'BPDay1Time2Sys343',
                   'BPDay1Time2Dia341', 'BPDay1Time2Dia342', 'BPDay1Time2Dia343',

                   'BPDay2Time1Sys341', 'BPDay2Time1Sys342', 'BPDay2Time1Sys343',
                   'BPDay2Time1Dia341', 'BPDay2Time1Dia342', 'BPDay2Time1Dia343',
                   'BPDay2Time2Sys341', 'BPDay2Time2Sys342', 'BPDay2Time2Sys343',
                   'BPDay2Time2Dia341', 'BPDay2Time2Dia342', 'BPDay2Time2Dia343']

        physical_health = pd.read_csv('data/datasets/ds004856/surveys/clean_physical_health.csv',
                                            index_col=0, usecols=keep_columns)

        physical_health_sys1 = physical_health[['BPDay1Time1Sys341', 'BPDay1Time2Sys341', 'BPDay2Time1Sys341', 'BPDay2Time2Sys341']].mean(axis=1)
        physical_health_sys2 = physical_health[['BPDay1Time1Sys342', 'BPDay1Time2Sys342', 'BPDay2Time1Sys342', 'BPDay2Time2Sys342']].mean(axis=1)
        physical_health_sys3 = physical_health[['BPDay1Time1Sys343', 'BPDay1Time2Sys343', 'BPDay2Time1Sys343', 'BPDay2Time2Sys343']].mean(axis=1)

        physical_health_dia1 = physical_health[['BPDay1Time1Dia341', 'BPDay1Time2Dia341', 'BPDay2Time1Dia341', 'BPDay2Time2Dia341']].mean(axis=1)
        physical_health_dia2 = physical_health[['BPDay1Time1Dia342', 'BPDay1Time2Dia342', 'BPDay2Time1Dia342', 'BPDay2Time2Dia342']].mean(axis=1)
        physical_health_dia3 = physical_health[['BPDay1Time1Dia343', 'BPDay1Time2Dia343', 'BPDay2Time1Dia343', 'BPDay2Time2Dia343']].mean(axis=1)

        physical_health =  pd.concat([physical_health_sys1, physical_health_dia1, physical_health_sys2, physical_health_dia2,
                                    physical_health_sys3, physical_health_dia3 ], axis=1)

        physical_health.columns = ['Sys1', 'Dia1', 'Sys2', 'Dia2', 'Sys3', 'Dia3']

        physical_health.to_csv('data/datasets/ds004856/surveys/physical_features.csv')

        return physical_health

    def load_clean_mental(self):
        keep_columns = ['S#',
                   'CESDTot371', 'CESDTot372', 'CESDTot373',
                   'ADASTot381', 'ADASTot382', 'ADASTot383']

        mental_health = pd.read_csv('data/datasets/ds004856/surveys/clean_mental_health.csv',
                                            index_col=0, usecols=keep_columns)
        mental_health2 = pd.read_csv('data/datasets/ds004856/surveys/clean_mental_health.csv',
                                            index_col=0, usecols=keep_columns)

        for wave in ['CESDTot371', 'CESDTot372', 'CESDTot373']:
            mental_health2[wave] = (mental_health[wave] >= 16).astype(int)

        for wave in ['ADASTot381', 'ADASTot382', 'ADASTot383']:
            mental_health2[wave] = (mental_health[wave] >= 10).astype(int)

        mental_health.columns = ['CESDepression1', 'CESDepression2', 'CESDepression3',
                                    'Alzheimer1', 'Alzheimer2', 'Alzheimer3']

        mental_health.to_csv('data/datasets/ds004856/surveys/mental_features.csv')

        return mental_health

    def load_clean_participants(self):
        cols = ['participant_id', 'AgeMRI_W1', 'AgeMRI_W2', 'AgeMRI_W3', 'Sex']
        participants_clean = pd.read_csv('data/datasets/ds004856/participants.tsv', sep='\t', index_col=0, usecols=cols)
        participants_clean.index = participants_clean.index.str.replace(r'sub-(\d+)', r'\1', regex=True)
        participants_clean.to_csv('data/datasets/ds004856/surveys/participants.csv')

    def join_patients_data(self):
        physical_health_clean = pd.read_csv('data/datasets/ds004856/surveys/physical_features.csv', index_col=0)
        mental_health_clean = pd.read_csv('data/datasets/ds004856/surveys/mental_features.csv', index_col=0)
        participants = pd.read_csv('data/datasets/ds004856/surveys/participants.csv', index_col=0)
        participants_data = pd.concat([participants, physical_health_clean, mental_health_clean], axis=1)

        w1 = participants_data[['AgeMRI_W1','Sex','Sys1','Dia1','CESDepression1','Alzheimer1']]
        w2 = participants_data[['AgeMRI_W2','Sex','Sys2','Dia2','CESDepression2','Alzheimer2']]
        w3 = participants_data[['AgeMRI_W3','Sex','Sys3','Dia3','CESDepression3','Alzheimer3']]

        for w in [w1, w2, w3]:
            w.columns = ['Age', 'Sex', 'Sys', 'Dia', 'CESDepression', 'Alzheimer']

        participants_data_sep = pd.concat([w1, w2, w3], axis=0)

        participants_data_sep['Participant'] = participants_data_sep.index
        participants_data_sep = participants_data_sep.dropna(subset=['Age'])
        participants_data_sep = participants_data_sep.reset_index(drop=True)

        participants_data_sep['Sex'] = participants_data_sep['Sex'].replace({'f': 0, 'm': 1}).astype(int)

        participants_data_sep = participants_data_sep[['Participant','Age','Sex','Sys','Dia','CESDepression','Alzheimer']]

        participants_data_sep.to_csv('data/datasets/ds004856/surveys/dataset.csv')


if __name__=="__main__":
    exit(0)