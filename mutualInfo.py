import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.feature_selection import mutual_info_classif


class mutualInformation:

    def __init__(self, phi, rsn, path):
        self.phi = phi  # Eigenvectors matrix
        self.rsn = rsn  # RSN binary matrix
        self.save_path = path
        self.mutual_info = None  # We'll store the result here after computation

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

        self.mutual_info = np.array(mutual_info)
        return self.mutual_info

    def plot_stemplot(self, subject, proj, RSN_names, name):
        """
        Generates and saves stem plots for each RSN.

        :param subject: subject ID
        :param proj: projection matrix (n_RSNs x num_harmonics)
        :param RSN_names: list of RSN names
        :param name: "mi" for mutual information or any other string for projection
        """
        save_folder = os.path.join(self.save_path, "images", f"{subject}_RSN")
        os.makedirs(save_folder, exist_ok=True)

        info = "mutual_information" if name == "mi" else "projection"
        y_label = "Mutual Information" if name == "mi" else "Projection"

        for i, rsn_name in enumerate(RSN_names):
            rsn_folder = os.path.join(save_folder, rsn_name, info)
            os.makedirs(rsn_folder, exist_ok=True)

            plt.figure(figsize=(6, 4))
            plt.stem(proj[i, 0:40])
            plt.title(f"{rsn_name} - {info}")
            plt.xlabel("N harmonic")
            plt.ylabel(y_label)

            image_path = os.path.join(rsn_folder, f"{rsn_name}_{info}.png")
            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            plt.close()

            print("Stem plot saved to", image_path)
