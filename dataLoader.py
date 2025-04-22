import os
import pandas as pd
import numpy as np
from computeFC import ComputeFC

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
        try:
            df = pd.read_csv(self.csv_path, header=None, sep=',', names=column_names, usecols=[0, 1])
        except Exception as e:
            print(f"Error reading CSV file at {self.csv_path}: {e}")
            return None

        return df

    def get_classification(self):
        """
        Gets a dictionary that indicates which patient is classified in each group
        :return: a dictionary with {subjID: groupLabel}
        """
        # Return a dictionary of subject IDs and their respective group labels (conditions)
        df = self._load_patient_data()
        if df is None:
            print("Patient data could not be loaded.")
            return None
        return dict(zip(df['id'], df['condition']))

    def get_group_avg_matrix(self, group="HC", sc=True):
        """
        Computes the average matrix (SC or FC) for all patients in a given group.
        :param group: Group label (HC, MCI, AD). Default is HC (control group).
        :param sc: If True, computes SC average; otherwise, computes FC average. Default is True
        :return: Mean matrix for the group.
        """
        # load the metadata for the patients
        df = self._load_patient_data()
        if df is None:
            print("Patient data could not be loaded.")
            return None
        subjects = df[df['condition'] == group]['id'].values  # get all subjects in the group=group

        # check if there are patients for the group, and if they have been loaded correctly
        if len(subjects) == 0:
            print(f"No subjects found for group {group}")
            return None

        # load first matrix to determine shape
        first_matrix = self.load_matrix(subjects[0], sc=sc)
        if first_matrix is None: # check if the matrix has been loaded
            print(f"Could not load matrix for subject {subjects[0]}")
            return None

        # initialize variables for accumulating results
        sum_matrix = np.zeros(first_matrix.shape)   # for SC
        valid_subjects = 0                          # for SC
        fc_matrices = []                            # for FC !!!

        # go through all the subjects
        for subject in subjects:
            matrix = self.load_matrix(subject, sc=sc) # load the matrix
            if matrix is not None:
                if sc: # sc==True
                    sum_matrix += matrix  # directly accumulate SC matrices
                    valid_subjects += 1   # count all the patients
                else:
                    # Calculate FC matrix (correlation of BOLD signals)
                    fc_matrix = ComputeFC.compute_from_fmri(matrix) # FC matrix = fMRI correlation
                    # we append the result into an array instead of summing matrices because correlations are NOT additive
                    fc_matrices.append(fc_matrix)

        # compute and return the average
        if sc:
            mean = sum_matrix / valid_subjects if valid_subjects > 0 else None      # compute the mean with the operation
        else:
            mean = np.mean(fc_matrices, axis=0) if len(fc_matrices) > 0 else None   # use the numpy mean() function

        return mean

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
        else:
            # load the fMRI data (BOLD signals) --> this is going to be processed !!!
            doc_name = f"{subject}_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt"
            file_path = os.path.join(self.base_folder,
                                     f'dades/fMRI/fMRI/{subject}/MNINonLinear/Results/Restingstate/', doc_name)
        data = pd.read_csv(file_path, sep=r'\s+', header=None)
        df = pd.DataFrame(data)

        return df.values  # numpy array !

    def get_all_fMRI(self, group='HC'):
        """
        Gets all the fMRI
        :param group: Group label (HC, MCI, AD). Default is HC (control group).
        :return: returns all the fMRI matrices in a list, for each subject
        """
        # load the metadata for the patients
        df = self._load_patient_data()
        if df is None:
            print("Patient data could not be loaded.")
            return None
        subjects = df[df['condition'] == group]['id'].values  # get all subjects in the group=group

        # check if there are patients for the group, and if they have been loaded correctly
        if len(subjects) == 0:
            print(f"No subjects found for group {group}")
            return None

        # load first matrix to determine shape
        first_matrix = self.load_matrix(subjects[0], sc=False) # loads fMRI
        if first_matrix is None:  # check if the matrix has been loaded
            print(f"Could not load matrix for subject {subjects[0]}")
            return None

        fMRI_matrices = []

        # go through all the subjects
        for subject in subjects:
            matrix = self.load_matrix(subject, sc=False)  # load the matrix
            if matrix is not None:
                    fMRI_matrices.append(matrix) # add the fMRI info to the matrix

        return fMRI_matrices