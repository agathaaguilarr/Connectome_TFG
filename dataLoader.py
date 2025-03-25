import os
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, base_folder):
        """
        Constructor.
        :param base_folder: base path to the data
        :param csv_path: path to the csv file where the information is stored
        """
        self.base_folder = base_folder
        self.csv_path = os.path.join(self.base_folder, "dades", "subjects.csv")
        self.patient_data = self._load_patient_data()

    def _load_patient_data(self):
        """
        Loads the patient data from the CSV file into a pandas DataFrame
        :return: a DataFrame containing the patient data (only 'id' and 'condition')
        """
        # Load the CSV file into a DataFrame, selecting only the 'id' and 'condition' columns
        column_names = ['id', 'condition']
        df = pd.read_csv(self.csv_path, header=None, sep=',', names=column_names, usecols=[0, 1])

        return df

    def get_classification(self):
        """
        Gets a dictionary that indicates which patient is classified in each group
        :return: a dictionary with {subjID: groupLabel}
        """
        # Return a dictionary of subject IDs and their respective group labels (conditions)
        df = self._load_patient_data()
        return dict(zip(df['id'], df['condition']))

    def load_matrix(self, subject):
        """
        loads the matrix for the patient specified by parameter
        :param subject --> the subject id (patient)
        :return: the dataframe (connectome information) for the correspondent patient
        """
        # construct the file path where the data file is located
        file_path = os.path.join(self.base_folder,
                                 f'dades/connectomes/connectomes/{subject}/DWI_processing/connectome_weights.csv')

        df = pd.read_csv(file_path, header=None, sep=' ')  # read the data from the CSV file located at file_path

        return df.to_numpy()  # converts the DataFrame df into a NumPy array

    def get_group_avg_matrix(self, group="HC"):
        """
        Computes the mean of the matrices for all the patients in the same group
        :param group: label for the group: normally it must be (HC, MCI, AD), by default is HC (control)
        :return: mean matrix for that group
        """
        df = self._load_patient_data()
        subjects = df[df['condition'] == group]['id'].values  # We get all the subjects that have HC condition (control)

        if len(subjects) == 0:
            print(f"No subjects found for group {group}")
            return None

        mm = self.load_matrix(subjects[0])  # Load the first subject’s matrix to determine the shape

        if mm is None:
            print(f"Could not load matrix for subject {subjects[0]}")
            return None

        sum_matrix = np.zeros(mm.shape)  # <-- Aquí está la corrección

        for subject in subjects:
            matrix = self.load_matrix(subject)  # Load the matrix
            if matrix is not None:
                sum_matrix += matrix

        return sum_matrix / len(subjects) if len(subjects) > 0 else None  # Compute the mean