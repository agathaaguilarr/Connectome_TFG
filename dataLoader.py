import os
import pandas as pd

class DataLoader:
    def __init__(self, base_folder):
        """
        constructor for this class
        :param base_folder --> the path to the base folder
        """
        self.base_folder = base_folder

    def load_matrix(self, subject):
        """
        loads the matrix for the patient specified by parameter
        :param subject --> the subject id (patient)
        :return: the dataframe (connectome information) for the correspondent patient
        """
        # construct the file path where the data file is located
        file_path = os.path.join(self.base_folder,
                                 f'dades/connectomes/connectomes/{subject}/DWI_processing/connectome_weights.csv')

        df = pd.read_csv(file_path, header=None, sep=' ') #read the data from the CSV file located at file_path

        return df.to_numpy() # converts the DataFrame df into a NumPy array
