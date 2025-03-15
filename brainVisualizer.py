
from matplotlib import cm, pyplot as plt
import os
import numpy as np

from project3DBrain import set_up_Glasser360_cortex

class BrainVisualizer:
    def __init__(self, work_folder, mode='Burbu'):
        """
        constructor for this class
        :param work_folder --> the path to the base folder
        :param mode --> it changes the way the images generates will be saved
        """
        self.work_folder = work_folder
        self.mode = mode
        self._setup()

    def _setup(self): # private function, meant to be used only inside this class!!!
        """
        this function dynamically configures the plotting utilities and cortical atlas setup
        """
        if self.mode == 'Burbu':
            import plot3DBrain_Utils as plt3Dbrain
        else:  # Gus
            import Utils.Plotting.plot3DBrain_Utils as plt3Dbrain

        self.plt3Dbrain = plt3Dbrain # the imported module is saved to be used after
        # the lambda function is an an anonymous function
        self.set_up_Glasser360_cortex = lambda: set_up_Glasser360_cortex(self.work_folder)

    def visualize(self, subject, harmonics, num_to_plot=360):
        """
        generates and saves 3D visualizations of brain harmonics for a given subject using the Glasser360 cortical atlas
        :param subject -->  identifier for the subject
        :param harmonics --> a matrix containing harmonic values (eigenvectors)
        :param num_to_plot --> the number of harmonics to visualize (limited by the number of available eigenvectors)
        all the harmonics by default (379)
        """
        save_dir = os.path.join(self.work_folder, "images", subject) # create a Directory to Save Images
        os.makedirs(save_dir, exist_ok=True)


        crtx = self.set_up_Glasser360_cortex() # the cortex model using the Glasser360 parcellation
        colors = cm.YlOrBr # colormap YlOrBr (Yellow-Orange-Brown) for visualization

        for i in range(min(num_to_plot, harmonics.shape[1])): # loop over harmonics
            # extract the first 360 elements (matching the number of cortical regions)
            harmonic = harmonics[:360, i]
            # normalizes the harmonic values to the range [0,1] for proper color scaling
            harmonic = (harmonic - np.min(harmonic)) / (np.max(harmonic) - np.min(harmonic))

            dataB = {'func_L': harmonic, 'func_R': harmonic} # data for harmonic values for both left and right hemispheres
            norm = plt.Normalize(vmin=np.min(harmonic), vmax=np.max(harmonic)) # defines a color normalization
            save_path = os.path.join(save_dir, f"harmonic{i}.png")

            # generate and Save the 3D Brain Visualization
            self.plt3Dbrain.multiview5(
                crtx, dataB, 360, colors,
                savePath=save_path,
                linear=True, lightingBias=0.1, mode='flatWire', shadowed=True,
                norm=norm
            )

    def visualize_RSN(self, subject, projectedRSN, names_vector, num_to_plot=360):
        """
        Generates and saves 3D visualizations of RSN (Resting State Networks) for a given subject using the Glasser360 cortical atlas
        :param subject --> identifier for the subject
        :param projectedRSN --> a matrix containing the RSN projection values
        :param names_vector --> list of RSN names for labeling
        :param num_to_plot --> the number of RSNs to visualize (limited by the number of available projections)
        """
        # create a Directory to Save Images
        save_dir = os.path.join(self.work_folder, "images", subject + "_RSN")
        os.makedirs(save_dir, exist_ok=True)

        crtx = self.set_up_Glasser360_cortex()  # set up the cortex model using the Glasser360 parcellation
        colors = cm.YlOrBr  # choose a colormap (Yellow-Orange-Brown) for visualization

        for i in range(min(num_to_plot, len(names_vector))):  # Loop over RSNs (or projections)
            # extract the first 360 elements (matching the number of cortical regions)
            rsn_projection = projectedRSN[:360, i]

            # normalize the RSN values to the range [0, 1] for proper color scaling
            rsn_projection = (rsn_projection - np.min(rsn_projection)) / (
                        np.max(rsn_projection) - np.min(rsn_projection))

            # prepare data for both left and right hemispheres
            dataB = {'func_L': rsn_projection, 'func_R': rsn_projection}
            norm = plt.Normalize(vmin=np.min(rsn_projection), vmax=np.max(rsn_projection))  # Define color normalization

            # set the save path for the image with the RSN labels
            save_path = os.path.join(save_dir, f"{names_vector[i]}_rsn.png")

            # generate and Save the 3D Brain Visualization
            self.plt3Dbrain.multiview5(
                crtx, dataB, 360, colors,
                savePath=save_path,
                linear=True, lightingBias=0.1, mode='flatWire', shadowed=True,
                norm=norm
            )
