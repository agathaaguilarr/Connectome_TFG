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

    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import mutual_info_classif

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

    def incremental_reconstruction(self, projection, RSN_vector):
        """
        Calculates the accumulated reconstruction error by summing weighted eigenvectors.

        :param projection: A vector of projected weights (shape: [N,])
        :param RSN_vector: DMN vector (shape: [N,])
        :return: Reconstruction errors list using Euclidean distance.
        """
        N = self.phi.shape[0]  # Number of brain regions

        # Define the percentages of eigenvectors used
        percentages = np.concatenate((
            np.array([0.0005, 0.005, 0.05]),  # Small values, similar to Fig.3 of S. Atasoy
            np.logspace(-2, 0, num=10)  # 10 values logarithmically spaced between 1% and 100%
        ))
        num_harmonics_list = (percentages * N).astype(int)  # Convert percentages to number of eigenvectors

        # Sort projection values by absolute magnitude (descending order)
        idx_sorted = np.argsort(-np.abs(projection))  # Get indices for sorting
        projection_sorted = projection[idx_sorted]  # Sort projection values
        phi_sorted = self.phi[:, idx_sorted]  # Sort eigenvectors accordingly

        errors = []

        for k in num_harmonics_list:  # Iterate through each quantity of eigenvectors (%)
            vec_reconstructed = np.zeros(N)  # Initialize the reconstruction vector

            for i in range(k):
                vec_reconstructed += projection_sorted[i] * phi_sorted[:, i]  # Accumulate reconstruction

            # Calculate the error using Euclidean distance
            error = np.linalg.norm(vec_reconstructed - RSN_vector)
            errors.append(error)

        # Normalize errors
        errors = np.array(errors) / np.max(errors)

        return errors, percentages
