import numpy as np
import pandas as pd

from scipy.stats.contingency import crosstab
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif

class Projecter:

    def __init__(self, e_vec):
        """
        constructor for this class
        :param phi: it is a nxm matrix (n>=m) where we have multiple basis vectors or an n vector
        """
        self.phi = e_vec  # self.phi = e_vec!!! the eigenvectors are constant in the projecter!

    def projectVector(self, x, invert=True):  # simplest way to project
        """
        projects a vector phi onto a single basis vector x using the
        dot product
        :param x --> one single basis n vector
        :param invert: Boolean flag to determine projection method.
                       - True: Considers both x and -x, keeping the strongest projection.
                       - False: Normal dot product.
        :return alpha --> the projected vector, its dimension is m
                """
        # alpha = np.dot(x, self.phi) # dot product
        if invert:
            alpha = round(max(np.dot(x.T, self.phi), np.dot(-x.T,self.phi)), 5)
        else:
            alpha = round(np.dot(x.T, self.phi), 5)

        return alpha


    def projectVectorRegion(self, x, invert=True):
        """
        Projects a vector or multiple vectors (phi) onto multiple basis vectors in x.

        :param x: An (n × m) matrix where columns are basis vectors.
        :param invert: Boolean flag to determine projection method.
                       - True: Considers both x and -x, keeping the strongest projection.
                       - False: Normal dot product.

        :return: A (m × k) matrix where each column corresponds to a projection of a phi vector.
        """
        x = np.atleast_2d(x)
        self.phi = np.atleast_2d(self.phi)

        min_rows = min(x.shape[0], self.phi.shape[0])
        x = x[:min_rows, :]
        phi = self.phi[:min_rows, :]

        print(x.shape)
        print(phi.shape)

        nr_modes = x.shape[1]  # Number of basis vectors (m)
        nr_phi = phi.shape[1]  # Number of phi vectors (k)

        alpha = np.zeros((nr_modes, nr_phi))  # Initialize alpha as (m × k)

        for i in range(nr_modes):
            for j in range(nr_phi):
                vec_x = x[:, i]  # (n,)
                vec_phi = phi[:, j]  # (n,)

                if invert:
                    proj = max(np.dot(vec_x.T, vec_phi), np.dot(-vec_x.T, vec_phi))
                else:
                    proj = np.dot(vec_x.T, vec_phi)
                alpha[i,j] = round(proj, 5)

        return alpha

    def projectVectorTime(self, F, invert=True):
        """
        projects a time-dependent vector (a function F(t)) onto multiple basis vectors over T time steps
            :param F --> a matrix nxT where n are the regions and T are the time steps
            :param invert --> which can be true or false (by default is true), used in the function projectVectorRegion
            :return alpha --> a matrix of projections of shape (k, T), where each column is the projection at a different time step
        """
        F = np.array(F)[:self.phi.shape[0]] # same dimensions as phi (e_Vec)
        n,T = F.shape
        nr_phi = self.phi.shape[1]  # Number of phi vectors (k)
        alpha = np.zeros((nr_phi, T))  # initialize a matrix to store the projections

        for t in range(T):
            F_t = F[:, t].reshape(-1, 1)  # compute F(t) for the current time step using the passed function, it also ensures the (n,1) shape, otherwise it will be (n,)
            alpha[:, t] = self.projectVectorRegion(F_t, invert)  # project F(t) and store the result
        return alpha