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

    def load_matrix(self, subject, sc=True):
        """
        Loads the matrix for a specific subject.

        :param subject: Subject ID (patient)
        :param sc: If True, loads Structural Connectivity (SC), otherwise loads fMRI time series.
        :return: NumPy array with the connectivity or fMRI data.
        """
        if sc:
            # load the structural matrix (SC) --> no need to process, it is directly the info we need
            file_path = os.path.join(self.base_folder,
                                     f'dades/connectomes/connectomes/{subject}/DWI_processing/connectome_weights.csv')
            df = pd.read_csv(file_path, header=None, sep=' ')  # Cargar como DataFrame
        else:
            # load the fMRI data (BOLD signals) --> this is going to be processed !!!
            doc_name = f"{subject}_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt"
            file_path = os.path.join(self.base_folder,
                                     f'dades/fMRI/fMRI/{subject}/MNINonLinear/Results/Restingstate/', doc_name)
            data = np.loadtxt(file_path)
            df = pd.DataFrame(data)

        return df.values  # numpy array !

    # def get_group_avg_matrix(self, group="HC", sc=True):
    #     """
    #     Computes the mean of the matrices for all the patients in the same group
    #     :param group: label for the group: normally it must be (HC, MCI, AD), by default is HC (control)
    #     :return: mean matrix for that group
    #     """
    #     df = self._load_patient_data()
    #     subjects = df[df['condition'] == group]['id'].values  # We get all the subjects that have HC condition (control)
    #
    #     if len(subjects) == 0:
    #         print(f"No subjects found for group {group}")
    #         return None
    #
    #     mm = self.load_matrix(subjects[0])  # Load the first subject’s matrix to determine the shape
    #
    #     if mm is None:
    #         print(f"Could not load matrix for subject {subjects[0]}")
    #         return None
    #
    #     sum_matrix = np.zeros(mm.shape)
    #
    #     for subject in subjects:
    #         matrix = self.load_matrix(subject)  # Load the matrix
    #         if matrix is not None:
    #             sum_matrix += matrix
    #
    #     return sum_matrix / len(subjects) if len(subjects) > 0 else None  # Compute the mean
    def get_group_avg_matrix(self, group="HC", sc=True):
        """
        Computes the average matrix for all patients in a given group.

        :param group: Group label (HC, MCI, AD). Default is HC (control group).
        :param sc: If True, computes SC average; otherwise, computes FC average.
        :return: Mean matrix for the group.
        """
        df = self._load_patient_data()
        subjects = df[df['condition'] == group]['id'].values  # Get all subjects in the group

        if len(subjects) == 0:
            print(f"No subjects found for group {group}")
            return None

        # load first matrix to determine shape
        first_matrix = self.load_matrix(subjects[0], sc=sc)

        if first_matrix is None:
            print(f"Could not load matrix for subject {subjects[0]}")
            return None

        # case 1: Structural Connectivity (SC) -> Direct Averaging
        if sc:
            sum_matrix = np.zeros(first_matrix.shape)
            valid_subjects = 0

            for subject in subjects:
                matrix = self.load_matrix(subject, sc=sc)
                if matrix is not None:
                    sum_matrix += matrix
                    valid_subjects += 1

            return sum_matrix / valid_subjects if valid_subjects > 0 else None

        # Case 2: Functional Connectivity (FC) -> Compute FC first, then Average
        else:
            fc_matrices = []
            for subject in subjects:
                bold_data = self.load_matrix(subject, sc=sc)
                if bold_data is not None:
                    fc_matrices.append(np.corrcoef(bold_data, rowvar=True))

            return np.mean(fc_matrices, axis=0) if len(fc_matrices) > 0 else None