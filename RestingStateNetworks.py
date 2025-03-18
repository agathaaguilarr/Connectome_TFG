import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class RestingStateNetworks:

    def __init__(self, path):
        self.RSN_labels_path = os.path.join(path, r"glasser360\Glasser360RSN_7_RSN_labels.csv")
        self.save_path = os.path.join(path, "images")

    def getRSNinformation(self):
        """"
        Loads RSN labels from a CSV file and creates binary vectors for each RSN.

        This function asks the user for a CSV file containing RSN labels for 360 brain regions.
        It extracts unique RSN names and generates a dictionary of binary vectors, where each
        vector has 1 for active regions and 0 otherwise.

        @:return RSN_vectors --> A dictionary where each RSN name maps to a (360,1) NumPy array showing active regions.
        """
        # open and read the CSV file into a DataFrame
        df = pd.read_csv(self.RSN_labels_path, header=None)

        if 'RSN' in df.columns:
            names = df['RSN'].tolist()  # get labels from the "RSN" column
        else:
            names = df.iloc[:, 0].tolist()  # get labels from the first column

        # get all the names without repetitions
        RSN_names = list(set(names))

        # DICTIONARY: creation of binary vectors for each RSN activity
        # each vector will have 360 positions, 1 where the region label corresponds (active) and 0 in the other cases
        RSN_matrix = np.zeros((len(names), len(RSN_names)), dtype=int)

        for i, label in enumerate(RSN_names):
            RSN_matrix[:, i] = np.array([1 if name == label
                                         else 0 for name in names])

        return RSN_matrix, RSN_names  # return de dictionary with all the RSN information (binary vectors for each RSN)

    def visualize_stemplot_RSN(self, subject, proj, RSN_names):
        """
        Generates and saves the stem plot for each RSN.

        :param subject: subject ID
        :param proj: projection matrix (7 x num_harmonics)
        :param RSN_names: a list for RSN names
        """
        save_folder = os.path.join(self.save_path, (subject + "_RSN"))
        os.makedirs(save_folder, exist_ok=True)  # Crear directorio si no existe

        for i, rsn_name in enumerate(RSN_names):
            plt.figure(figsize=(6, 4))
            plt.stem(proj[i, 1:51])
            plt.title(f"{rsn_name} projection")
            plt.xlabel("N harmonic")
            plt.ylabel("Projection value")

            image_path = os.path.join(save_folder, f"{rsn_name}_projection.png")
            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            plt.close()
            print("Stem plot saved to " + image_path)


