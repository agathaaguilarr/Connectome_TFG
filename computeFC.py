import numpy as np

class computeFC():
    def __init__(self, ):
        self.input = input
        self.output = output

    def compute_from_fmri(self, fmri):
        """
        computes the Functional Connectivity Matrix from FMRI, which is no more than the correlation of the BOLD (fMRI) data
        :param fmri: BOLD data
        :return: the Functional Connectivity Matrix (cc)
        """
        FC = np.corrcoef(fmri, rowvar=True)  # Pearson correlation coefficients
        return FC