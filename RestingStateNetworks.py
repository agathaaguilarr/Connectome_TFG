import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class RestingStateNetworks:
    def __init__(self, path, save_path):
        self.RSN_labels_path = path
        self.save_path = save_path

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

    def visualize_stemplot_RSN(self, subject, proj, RSN_names, name):
        """
        Generates and saves the stem plot for each RSN.

        :param subject: subject ID
        :param proj: projection matrix (7 x num_harmonics)
        :param RSN_names: a list for RSN names
        """
        save_folder = os.path.join(self.save_path, (subject + "_RSN"))
        os.makedirs(save_folder, exist_ok=True)

        if name == "mi":
            info = "mutual_information"
            y_label = "mutual information"
        else:
            info = "projection"
            y_label = "projection"

        for i, rsn_name in enumerate(RSN_names):
            # create folders and directories
            rsn_folder = os.path.join(save_folder, rsn_name, info)
            os.makedirs(rsn_folder, exist_ok=True)

            #create the plot
            plt.figure(figsize=(6, 4))
            plt.stem(proj[i, 1:51])
            plt.title(f"{rsn_name} {info}")
            plt.xlabel("N harmonic")
            plt.ylabel(y_label)

            # path to save the final image
            image_path = os.path.join(rsn_folder, f"{rsn_name}_{info}.png")
            plt.savefig(image_path, dpi=300, bbox_inches="tight")
            plt.close()
            print("Stem plot saved to " + image_path)

    def plot_reconstruction_error(self, errors, mode):
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
        save_path = os.path.join(save_dir, "Reconstruction_Error.png")

        # plot the reconstruction error
        plt.figure(figsize=(8, 5))
        plt.plot(errors, marker='o', linestyle='-')
        plt.xlabel('% used eigenvectors')
        plt.ylabel('Reconstruction error')
        plt.title('Reconstruction Error for Default Mode (DMN)')
        plt.grid()

        # save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Reconstruction error plot saved")

    def get_desired_RSN(self, RSN_names, RSN_dictionary, proj, name):
        # get default mode information for the reconstruction error
        if name in RSN_names:
            dmn_index = RSN_names.index(name)
            desired_RSN = RSN_dictionary[:, dmn_index]  # get only the original DMN
            desired_proj = proj[dmn_index, :]  # get the DMN projection
        else:
            raise ValueError("The ", name, " RSN is not on the dictionary.")

        return desired_RSN, desired_proj