import numpy as np
import pandas as pd

class Projecter:

    def __init__(self, phi):
        """
        constructor for this class
        :param phi: it is a nxm matrix (n>=m) where we have multiple basis vectors or an n vector
        """
        self.phi = phi

    def projectVector(self, x): # simplest way to project
        """
        projects a vector phi onto a single basis vector x using the dot product
            :param x --> one single basis n vector
            :return alpha --> the projected vector, its dimension is m
        """
        alpha = np.dot(x, self.phi) # dot product
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

        nr_modes = x.shape[1]  # Number of basis vectors (m)
        print("Number of basis vectores ", nr_modes)
        nr_phi = self.phi.shape[1]  # Number of phi vectors (k)
        print("Dimensions for phi (harmonics): ", self.phi.shape)

        alpha = np.zeros((nr_modes, nr_phi))  # Initialize alpha as (m × k)

        for i in range(nr_modes):
            for j in range(nr_phi):
                vec_x = x[:, i]  # (n,)
                vec_phi = self.phi[:, j]  # (n,)

                # vec_x = x.reshape(-1, 1)  # Asegurar que sea (360, 1)
                # vec_phi = self.phi.reshape(-1, 1)  # Asegurar que sea (360, 1)

                # vec_phi = self.phi
                # vec_x = x

                if invert:
                    alpha[i, j] = round(max(np.dot(vec_x.T, vec_phi), np.dot(-vec_x.T, vec_phi)),5)
                else:
                    alpha[i, j] = round(np.dot(vec_x.T, vec_phi),5)

        return alpha

    def projectVectorTime(self, F, invert=True):
        """
        projects a time-dependent vector (a function F(t)) onto multiple basis vectors over T time steps
            :param F --> a matrix nxT where n are the regions and T are the time steps
            :param invert --> which can be true or false (by default is true), used in the function projectVectorRegion
            :return alpha --> a matrix of projections of shape (k, T), where each column is the projection at a different time step
        """
        n,T = F.shape
        nr_phi = self.phi.shape[1]  # Number of phi vectors (k)
        alpha = np.zeros((nr_phi, T))  # initialize a matrix to store the projections

        for t in range(T):
            F_t = F[:, t].reshape(-1, 1)  # compute F(t) for the current time step using the passed function, it also ensures the (n,1) shape, otherwise it will be (n,)
            alpha[:, t] = self.projectVectorRegion(F_t, invert)  # project F(t) and store the result
        return alpha

