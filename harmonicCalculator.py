import numpy as np
from laplacianCalculator import LaplacianCalculator, SymmetricLaplacian

class HarmonicCalculator:
    def __init__(self, th=0.0001): #constructor
        """
        constructor for this class
        :param th --> the threshold on the parameter is the one by default in case there is no th defined when the constructor is created
        """
        self.th = th

    def compute_harmonics(self, M):
        """
        computes the harmonics (eig vec i val) given a matrix M
        :param M --> a matrix nxn
        :return: eigen vectors i values (real part) of that matrix
                 e_vec --> an nxn matrix
                 e_val --> an n vector
        """
        lp = LaplacianCalculator() # instance the laplacianCalculator
        lps = SymmetricLaplacian() # instance the symmetricLaplacian

        A = lp.get_adj(M, self.th) # get the adjacency matrix
        D = lp.get_deg(A)                 # get the degree matrix
        L = lps.get_laplacian(A, D)       # get the symmetric laplacian

        # compute the eigen vectors and eigen values, they are given in a ascendent way
        # from smaller eigenvalue (more important, more information) to greater eigenvalue (less important)
        # compute eigenvalues and eigenvevctors (the function eigh already assumes the matrix symmetric)
        e_val, e_vec = np.linalg.eigh(L)

        # although they are already sorted from smaller to greater, we do it again to assure it!
        idx = np.argsort(e_val)
        e_val = e_val[idx]
        e_vec = e_vec[:, idx]

        return e_val, np.real(e_vec)