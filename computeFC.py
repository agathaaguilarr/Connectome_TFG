import numpy as np


class ComputeFC:
    @staticmethod  # it makes possible to call the functions without instantiating the class
    def compute_from_fmri(fmri):
        """
        Computes the Functional Connectivity Matrix from fMRI data.

        :param fmri: 2D NumPy array (BOLD signals, shape: [n_regions, timepoints])
        :return: Functional Connectivity Matrix (FC), shape: [n_regions, n_regions]
        """
        return np.corrcoef(fmri, rowvar=True)  # Pearson correlation
