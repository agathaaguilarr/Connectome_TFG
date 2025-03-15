import numpy as np
import pandas as pd

class RestingStateNetworks:

    def __init__(self, path):
        self.path = path

    def getRSNinformation(self):
        """"
        Loads RSN labels from a CSV file and creates binary vectors for each RSN.

        This function asks the user for a CSV file containing RSN labels for 360 brain regions.
        It extracts unique RSN names and generates a dictionary of binary vectors, where each
        vector has 1 for active regions and 0 otherwise.

        @:return RSN_vectors --> A dictionary where each RSN name maps to a (360,1) NumPy array showing active regions.
        """
        # open and read the CSV file into a DataFrame
        df = pd.read_csv(self.path, header=None)
        print(df.shape)

        if 'RSN' in df.columns:
            names = df['RSN'].tolist()  # get labels from the "RSN" column
        else:
            names = df.iloc[:, 0].tolist()  # get labels from the first column

        # get all the names without repetitions
        RSN_names = list(set(names))
        print("RSN únicos:", RSN_names)

        # DICTIONARY: creation of binary vectors for each RSN activity
        # each vector will have 360 positions, 1 where the region label corresponds (active) and 0 in the other cases
        RSN_matrix = np.zeros((len(names), len(RSN_names)), dtype=int)

        for i, label in enumerate(RSN_names):
            RSN_matrix[:, i] = np.array([1 if name == label
                                         else 0 for name in names])

        return RSN_matrix, RSN_names  # return de dictionary with all the RSN information (binary vectors for each RSN)