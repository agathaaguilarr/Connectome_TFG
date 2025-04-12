import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import mutual_info_classif


class mutualInformation:

    def __init__(self, phi, rsn, path):
        self.phi = phi
        self.rsn = rsn  # RSN_matrix (binary matrix)
        self.save_path = path

    def computeMutualInformation(self):
        """
        Computes the mutual information between RSNs and each eigenvector.
        :return: mutual_info: (RSNs x harmonics) matrix of mutual information
        """
        mutual_info = []
        for i in range(self.rsn.shape[1]):  # iterate through RSNs
            vector = self.rsn[:, i]
            mi_rsn_i = mutual_info_classif(self.phi, vector, discrete_features=False)
            mutual_info.append(mi_rsn_i)

        return np.array(mutual_info)
