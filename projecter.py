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

    def projectVector(self, x, invert=False):  # simplest way to project
        """
        projects a vector phi onto a single basis vector x using the
        dot product
        :param x --> one single basis n vector
        :return alpha --> the projected vector, its dimension is m
                """
        # alpha = np.dot(x, self.phi) # dot product
        if invert:
            alpha = round(max(np.dot(x.T, self.phi), np.dot(-x.T,self.phi)), 5)
        else:
            alpha = round(np.dot(x.T, self.phi), 5)

        return alpha


    def projectVectorRegion(self, x, invert=False):
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
        #print("Number of basis vectores ", nr_modes)
        nr_phi = self.phi.shape[1]  # Number of phi vectors (k)
        #print("Dimensions for phi (harmonics): ", self.phi.shape)

        alpha = np.zeros((nr_modes, nr_phi))  # Initialize alpha as (m × k)

        for i in range(nr_modes):
            for j in range(nr_phi):
                vec_x = x[:, i]  # (n,)
                vec_phi = self.phi[:, j]  # (n,)

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

    def computeMutualInformation(self, RSN):
        mutual_info = []
        for rsn in range(RSN.shape[1]):  # each iteration, one different RSN
            vector = RSN[:, rsn]

            mi_rsn_i = mutual_info_classif(self.phi, vector, discrete_features=False)

            mutual_info.append(mi_rsn_i)

        return np.array(mutual_info)  # MI matriz: rows = RSN, columns = eigenvectors

        # FORMAT OF THE FINAL RESULT:
            # [[0.1, 0.3, 0.05, 0.2],  # MI between RSN 0 and all the eigenvectors
            #                        .................
            #                        .................
            #  [0.2, 0.5, 0.3, 0.1]]   # MI between RSN 6 and all the eigenvectors

    def sort_projections(self, projection):

        # order the eigenvectors (phi) and its projections (projection) depending on the projection value (weights), the order is descendent
        # order the magnitude of the projection from bigger to smaller and get the indices
        sorted_indices = np.argsort(np.abs(projection))[::-1]
        # order the eigenvectors (self.phi) and the projections (projection) depending on the ordered indices from before
        sorted_projection = projection[sorted_indices]
        sorted_phi = self.phi[:, sorted_indices]

        return sorted_projection, sorted_phi

    def reconstruction_error(self, projection, RSN_vector, phi_partial):
        """
            Calculates the accumulated reconstruction error by summing weighted eigenvectors.

            :param projection: A vector of projected weights (shape: [N,])
            :param RSN_vector: RSN vector (shape: [N,])
            :return: Normalized reconstruction errors list
        """

        # unification of dimensions, taking as reference the eigenvector matrix size
        n = phi_partial.shape[0]
        projection = projection[:n]
        phi = phi_partial[:n, :]

        # calculate the reconstructed vector (multiplying each vector i to the correspondent projection i (value))
        reconstructed_vector = np.dot(phi, projection)

        # calculate the reconstruction error using the Euclidean distance between the real RSN vector and the reconstructed one
        error = np.linalg.norm(RSN_vector - reconstructed_vector)

        return error

    def accumulated_reconstruction_error(self, projection, RSN_vector):

        proj, phi = self.sort_projections(projection)

        error_accumulated = []  # we will save here the errors

        total_components = RSN_vector.shape[0]  # number of total components

        for i in range(1, total_components + 1):  # Iteramos sobre los componentes
            # we take the first i components, each time i will get grater until all the components are taken into account
            proj_partial = proj[:i]  # % of projectionses
            phi_partial = phi[:, :i]  # % of eigenvectors

            # calculate the reconstructed error for the selected portion
            error = self.reconstruction_error(proj_partial, RSN_vector, phi_partial)
            error_accumulated.append(error)  # save the error

        # normalize between 0-1
        norm_acc_error = np.array(error_accumulated)/max(np.array(error_accumulated))
        return norm_acc_error