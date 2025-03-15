# import my Classes
from dataLoader import DataLoader
from harmonicCalculator import HarmonicCalculator
from brainVisualizer import BrainVisualizer
from projecter import Projecter
from RestingStateNetworks import RestingStateNetworks

# other util imports
import numpy as np

Mode = "Burbu"

class Pipeline:

    #MAIN GOAL: Convert raw connectivity data into meaningful harmonic representations and visualizations.

    def __init__(self, work_folder):
        """
        constructor for this class
        :param work_folder --> the path to the working folder
        """
        self.work_folder = work_folder
        self.data_loader = DataLoader(self.work_folder) # initializes the DataLoader, to load the information from the work_folder
        self.harmonic_calculator = HarmonicCalculator(th=0.00065) # initialize an instance of the HarmonicCalculator class
        self.visualizer = BrainVisualizer(self.work_folder, Mode) # creates an instance of BrainVisualizer to generate plots

    def run(self, subject):
        """
        Processes the Structural Connectivity (SC) matrix for a given subject by computing its harmonic components
        and generating 3D brain visualizations.

        Steps:
        1. Loads the SC matrix for the specified subject.
        2. Normalizes the SC matrix by scaling it and removing self-connections.
        3. Computes the eigenvalues and eigenvectors (harmonics) from the normalized SC matrix.
        4. Generates and saves 3D brain visualizations using the computed eigenvectors to get the needed information.

        :param subject
        """
        print('Loading matrix...')
        # loads Structural Connectivity for the subject specified
        SC = self.data_loader.load_matrix(subject)

        # normalize the Structural Connectivity matrix
        M = (SC / np.max(SC)) * 1.0
        M -= np.diag(np.diag(M))

        print('Computing harmonics...')
        # compute the harmonics (eigen vectors) for the SC normalized matrix, also the eigen values, which won't be used for now...
        e_val, e_vec = self.harmonic_calculator.compute_harmonics(M)
        print("Dimensiones e_vec (harmonics): ", e_vec.shape)

        print("Getting the RSN dictionary")
        rsn_path = r"C:\Users\AGATHA\Desktop\4t_GEB\TFG\glasser360\Glasser360RSN_7_RSN_labels.csv"
        rsnInfo = RestingStateNetworks(rsn_path)
        RSN_dictionary, RSN_names = rsnInfo.getRSNinformation()
        print(RSN_names)
        print(RSN_dictionary)
        print("RSN_dictionary shape:", RSN_dictionary.shape)  # Debería ser (360, 7)

        print("Projecting RSN information...")
        e_vec = e_vec[:RSN_dictionary.shape[0], :]
        print("Dimensions post [:RSN_dictionary.shape[0], :]: ", e_vec.shape)
        projecter = Projecter(e_vec)
        proj = projecter.projectVectorRegion(RSN_dictionary)
        print(proj)
        print("RSN Projection: ", proj.shape)

        self.visualizer.visualize_RSN(subject, proj.T, RSN_names)

        # print('Generating and saving Brain3D images...')
        # # generate the 3D brain images (plots) for the subject specified
        # self.visualizer.visualize(subject, e_vec)
        # print('Image generation is done!')

    def get_work_folder(self):
        """
        get the path to the working folder
        :return work_folder --> path to the working folder
        """
        return self.work_folder


if __name__ == '__main__':
    if Mode == "Burbu":
        work_folder = "C:/Users/AGATHA/Desktop/4t_GEB/TFG/"
    else:
        work_folder = '../../_Data_Raw/ADNI-A/'

    pipeline = Pipeline(work_folder) # initialize the pipeline, which will work with the work_folder
    pipeline.run('002_S_5178') # runs the pipeline with the subject specified by parameter
    print('The whole tasks have been done!!!')