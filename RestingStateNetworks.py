import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#REVISAR!!!

class RestingStateNetworks:
    def __init__(self, path, save_path):
        self.RSN_labels_path = path
        self.save_path = save_path

    def getRSNinformation(self):
        """
        Loads RSN labels from a CSV file and creates binary vectors for each RSN.

        :return: RSN_matrix -> (360, n) matrix with binary vectors for each RSN
                 RSN_names -> list of RSN names
        """
        # Load CSV
        df = pd.read_csv(self.RSN_labels_path, header=None)

        # Extract RSN names
        if 'RSN' in df.columns:
            names = df['RSN'].tolist()
        else:
            names = df.iloc[:, 0].tolist()

        RSN_names = sorted(list(set(names)))  # Sorted for consistent order

        # Create binary matrix
        RSN_matrix = np.zeros((len(names), len(RSN_names)), dtype=int)
        for i, label in enumerate(RSN_names):
            RSN_matrix[:, i] = [1 if name == label else 0 for name in names]

        return RSN_matrix, RSN_names

    def get_desired_RSN(self, RSN_names, RSN_dictionary, proj, name="Default"):
        """
        Extracts information for a specific RSN.

        :param RSN_names: list of RSN names
        :param RSN_dictionary: binary matrix (360 x n_RSNs)
        :param proj: projection matrix (n_RSNs x num_harmonics)
        :param name: name of the RSN to extract
        :return: desired_RSN -> binary vector (360,)
                 desired_proj -> projection vector (num_harmonics,)
        """
        if name in RSN_names:
            index = RSN_names.index(name)
            desired_RSN = RSN_dictionary[:, index]
            desired_proj = proj[index, :]
            return desired_RSN, desired_proj
        else:
            raise ValueError(f"The RSN '{name}' is not in the dictionary.")
