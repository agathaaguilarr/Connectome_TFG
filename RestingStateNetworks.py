import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#REVISAR!!!

class RestingStateNetworks:
    def __init__(self, path, save_path):
        self.RSN_labels_path = path
        self.save_path = save_path

    def getRSNinformation(self):
        """
        Loads RSN labels from a CSV file and creates binary vectors for each RSN.

        :return: RSN_matrix -> (360, n) matrix with binary vectors for each RSN
                 RSN_names -> list of RSN names
        """
        # Load CSV
        df = pd.read_csv(self.RSN_labels_path, header=None)

        # Extract RSN names
        if 'RSN' in df.columns:
            names = df['RSN'].tolist()
        else:
            names = df.iloc[:, 0].tolist()

        RSN_names = sorted(list(set(names)))  # Sorted for consistent order

        # Create binary matrix
        RSN_matrix = np.zeros((len(names), len(RSN_names)), dtype=int)
        for i, label in enumerate(RSN_names):
            RSN_matrix[:, i] = [1 if name == label else 0 for name in names]

        return RSN_matrix, RSN_names

    def visualize_stemplot_RSN(self, subject, proj, RSN_names, name):
        """
        Generates and saves stem plots for each RSN.

        :param subject: subject ID
        :param proj: projection matrix (n_RSNs x num_harmonics)
        :param RSN_names: list of RSN names
        :param name: "mi" for mutual information or any other string for projection
        """
        save_folder = os.path.join(self.save_path, f"{subject}_RSN")
        os.makedirs(save_folder, exist_ok=True)

        info = "mutual_information" if name == "mi" else "projection"
        y_label = "Mutual Information" if name == "mi" else "Projection"

        for i, rsn_name in enumerate(RSN_names):
            rsn_folder = os.path.join(save_folder, rsn_name, info)
            os.makedirs(rsn_folder, exist_ok=True)

            plt.figure(figsize=(6, 4))
            plt.stem(proj[i, 1:51], use_line_collection=True)
            plt.title(f"{rsn_name} - {info}")
            plt.xlabel("N harmonic")
            plt.ylabel(y_label)

            image_path = os.path.join(rsn_folder, f"{rsn_name}_{info}.png")
            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            plt.close()

            print("Stem plot saved to", image_path)

    def get_desired_RSN(self, RSN_names, RSN_dictionary, proj, name="Default"):
        """
        Extracts information for a specific RSN.

        :param RSN_names: list of RSN names
        :param RSN_dictionary: binary matrix (360 x n_RSNs)
        :param proj: projection matrix (n_RSNs x num_harmonics)
        :param name: name of the RSN to extract
        :return: desired_RSN -> binary vector (360,)
                 desired_proj -> projection vector (num_harmonics,)
        """
        if name in RSN_names:
            index = RSN_names.index(name)
            desired_RSN = RSN_dictionary[:, index]
            desired_proj = proj[index, :]
            return desired_RSN, desired_proj
        else:
            raise ValueError(f"The RSN '{name}' is not in the dictionary.")
