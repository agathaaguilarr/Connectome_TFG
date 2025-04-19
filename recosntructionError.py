import numpy as np
import matplotlib.pyplot as plt
import os

class reconstructionError:

    def __init__(self, phi):
        self.phi = phi

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

    def accumulated_reconstruction_error(self, projection, RSN_vector, sort=False):
        """
        it computes the accumulated reconstruction error
        :param projection: the projection corresponding to RSN_vector
        :param RSN_vector: the original RSN vector
        :param sort: whether to sort the reconstruction error or not
        :param name: name of the RSNa
        :return norm_acc_error: the normalized incremental reconstruction error
        """
        if sort:  # sort in a descendent order
            proj, phi = self.sort_projections(projection) # descendent sorting of the projections (weights) and the eigenvectors (harmonics)
            # the projections tell us the importance of each eigenvector in the RSN reconstruction
        else:  # don't sort
            proj = projection
            phi = self.phi

        # prepare instances
        error_accumulated = []  # we will save here the errors
        total_components = min(self.phi.shape[1], RSN_vector.shape[0])

        for i in range(1, total_components + 1):
            # each time we take the first i components, each time i will get grater (+1) until all the components are taken into account
            proj_partial = proj[:i]  # % of projections
            phi_partial = phi[:, :i]  # % of eigenvectors

            # calculate the reconstructed error for the selected portion
            error = self.reconstruction_error(proj_partial, RSN_vector,
                                              phi_partial)  # at each iteration we send 1 more pair of (eigenvector - projection)
            error_accumulated.append(error)  # save the error

        # normalize between 0-1
        norm_acc_error = np.array(error_accumulated) / max(np.array(error_accumulated))
        return norm_acc_error

    def sort_projections(self, projection):
        """
        order the eigenvectors (self.phi) and its projections (projection) depending on the projection value (weights/importance),
        the order is descendent, from grater to smaller
        :param projection: an n vector corresponding to the eigenvectors projected to a vector (normally RSN)
        :return: sorted_projection, sorted_phi, respectively the ordered projections and eigenvectors
        """
        # order the magnitude of the projection from bigger to smaller and get the indices
        sorted_indices = np.argsort(np.abs(projection))[::-1]
        # order the eigenvectors (self.phi) and the projections (projection) depending on the ordered indices from before
        sorted_projection = projection[sorted_indices]
        sorted_phi = self.phi[:, sorted_indices]

        return sorted_projection, sorted_phi

    def plot_all_reconstruction_errors(self, errors_dict, mode, s):
        """
        Plots the reconstruction error for multiple RSNs in the same figure with different colors.

        :param errors_dict: Dictionary where keys are RSN names and values are error arrays.
        :param mode: Mode used for saving the plots.
        """

        if mode == "Burbu":
            save_dir = os.path.join(r"C:\Users\AGATHA\Desktop\4t_GEB\TFG\images\rec_errors", s)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = os.path.join('./_Results', s)
        os.makedirs(save_dir, exist_ok=True)  # Crea la carpeta si no existe

        plt.figure(figsize=(10, 6))

        # colors for the different plots
        colors = plt.cm.viridis(np.linspace(0, 1, len(errors_dict)))

        # plot each RSN
        for (rsn, errors), color in zip(errors_dict.items(), colors):
            plt.plot(errors, label=rsn, color=color)

        plt.xlabel("% used vectors")
        plt.ylabel("Normalized reconstruction error")
        plt.title("RECONSTRUCTION ERROR FOR EACH RSN")
        plt.legend(title="RSNs", fontsize='small', loc='best')
        plt.grid(True)

        #save the plot
        save_path = os.path.join(save_dir, "All_RSN_Reconstruction_Errors.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"All reconstruction error plot saved at {save_path}")

    def plot_reconstruction_error(self, errors, mode, rsn, sort=False):
        """
        Plots the reconstruction error for Default Mode RSN

        :param errors: normalized reconstruction error
        :param percentages: percentages used to calculate the reconstruction error
        """

        if mode == "Burbu":
            save_dir = r"C:\Users\AGATHA\Desktop\4t_GEB\TFG\images\mean_HC_RSN"
        else:
            save_dir = './_Results'
        os.makedirs(save_dir, exist_ok=True)  # creates the folder if not existing
        if sort:
            save_path = os.path.join(save_dir, rsn, "_Sorted_Reconstruction_Error.png")
        else:
            save_path = os.path.join(save_dir, rsn, "_Reconstruction_Error.png")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # plot the reconstruction error
        plt.figure(figsize=(8, 5))
        plt.plot(errors, marker='o', linestyle='-')
        plt.xlabel('% used eigenvectors')
        plt.ylabel('Reconstruction error')
        plt.title('Reconstruction Error for Default Mode (DMN)')
        plt.grid()