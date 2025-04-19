
import numpy as np

class LaplacianCalculator: #superclass
    @staticmethod
    def get_adj(M, th):
        """
        get the binary adjacency matrix based on a threshold
        :param M: adjacency nxn matrix
        :param th: threshold, it is defined by the user before (or it is 0.0005 by default)
        :return: binary adjacency nxn matrix
        """
        A = np.copy(M) # do not binarize the matrix ( to binarize change this line to: np.ones(M.shape) )
        A[A <= th] = 0
        A = np.maximum(A, A.T)  # ensure symmetry

        return A

    @staticmethod
    def get_deg(A):
        """
        calculates the degree of the binary adjacency matrix
        :param A: binary adjacency nxn matrix
        :return: degree of the binary nxn adjacency matrix
        """
        deg = np.sum(A, axis=0)
        return np.diag(deg)

    def get_laplacian(self, A, D):
        """
        calculates the not normalized laplacian --> L=D−A
        :param A --> adjacency nxn matrix
        :param D --> degree nxn matrix
        :return: L --> the unormalized (and unsymmetrical) Laplacian, a nxn matrix
        """
        return D - A

class UnnormalizedLaplacian(LaplacianCalculator):  # Subclass
    def get_laplacian(self, A, D):
        """
        calculates the unnormalized (and unsymmetrical) Laplacian from the superclass!
        :param A --> adjacency nxn matrix
        :param D --> degree nxn matrix
        :return L --> L --> the unormalized (and unsymmetrical) Laplacian, a nxn matrix
        """

        L = super().get_laplacian(A, D)
        return L


class SymmetricLaplacian(LaplacianCalculator): #subclasse

    def get_laplacian(self, A, D):
        """
        calculates the symmetric laplacian --> L_sym = D^(−1/2)·L·D^(−1/2)
        :param A --> adjacency nxn matrix
        :param D --> degree nxn matrix
        :return L_sym --> the symmetric laplacian, a nxn matrix
        """

        L = super().get_laplacian(A, D) # it calls this method from the parent class
        N = A.shape[0] # we get the nodes in the graph (number of rows on the adjacency matrix)
        D2 = D

        for i in range(N): # iterate all over the graph nodes

            if D2[i, i] > 0: # if the degree (value) is greater than 0, we can compute the invert of the squareroot
                D2[i, i] = 1 / np.sqrt(D2[i, i]) #  invert of the squareroot --> D2[i, i]^(-0.5)

            else: # we cannot divide something by 0, so we have to take it into account
                D2[i, i] = 0 # all the diagonal (same region connected with same region) will still be 0

        L_sym = D2 @ L @ D2 # we compute the symmetric laplacian
        return L_sym