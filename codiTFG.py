# import my Classes
from harmonicCalculator import HarmonicCalculator
from brainVisualizer import BrainVisualizer
from projecter import Projecter
from RestingStateNetworks import RestingStateNetworks
import os

# other util imports
import numpy as np

Mode = "Burbu"  # Burbu / Gus
type = "mean"


if Mode == "Gus":
    import DataLoaders.WorkBrainFolder as WBF
    import DataLoaders.ADNI_A as ADNI_A
else:
    from dataLoader import DataLoader


class Pipeline:

    #MAIN GOAL: Convert raw connectivity data into meaningful harmonic representations and visualizations.

    def __init__(self, work_folder):
        """
        constructor for this class
        :param work_folder --> the path to the working folder
        """
        self.work_folder = work_folder
        if Mode == "Gus":
            self.data_loader = ADNI_A.ADNI_A()
        else:
            self.data_loader = DataLoader(self.work_folder) # initializes the DataLoader, to load the information from the work_folder
        self.harmonic_calculator = HarmonicCalculator(th=0.00065) # initialize an instance of the HarmonicCalculator class
        self.visualizer = BrainVisualizer(self.work_folder, Mode) # creates an instance of BrainVisualizer to generate plots

    def run_structural_connectivity(self, SC, subject):
        """
        Processes the Structural Connectivity (SC) matrix for a given subject by computing its harmonic components
        and generating 3D brain visualizations.
        """

        # Normalize SC matrix
        M = (SC / np.max(SC)) * 1.0
        M -= np.diag(np.diag(M))

        print('Computing harmonics...')
        e_val, e_vec = self.harmonic_calculator.compute_harmonics(M)
        print("Dimensiones e_vec (harmonics): ", e_vec.shape)

        print("Getting the RSN dictionary")
        if Mode == "Burbu":
            parc_path = self.work_folder + r"glasser360/"
            save_path = os.path.join(self.work_folder, "images")
        else:
            parc_path = WBF.WorkBrainProducedDataFolder + '_Parcellations/'
            save_path = './_Results/'

        rsn_path = parc_path + 'Glasser360RSN_7_RSN_labels.csv'
        rsnInfo = RestingStateNetworks(rsn_path, save_path)
        RSN_dictionary, RSN_names = rsnInfo.getRSNinformation()
        print("RSN_dictionary shape:", RSN_dictionary.shape)

        print("Projecting RSN information...")
        e_vec = e_vec[:RSN_dictionary.shape[0], :]
        projecter = Projecter(e_vec)  # self.phi = e_vec!!!
        proj = projecter.projectVectorRegion(RSN_dictionary)  # project all the RSN

        # get default mode information for the reconstruction error
        if "Default" in RSN_names:
            dmn_index = RSN_names.index("Default")
            RSN_DMN = RSN_dictionary[:, dmn_index]  # get only the original DMN
            proj_DMN = proj[dmn_index, :]  # get the DMN projection
        else:
            raise ValueError("La Default Mode Network (DMN) no está en el diccionario de RSNs.")

        print("Calculating Mutual Information between RSN and eigenvectors...")
        mi_matrix = projecter.computeMutualInformation(RSN_dictionary)

        # calculate reconstruction errors
        errors, percentages = projecter.incremental_reconstruction(proj_DMN, RSN_DMN)

        # visualize every plot
        rsnInfo.plot_reconstruction_error(errors,percentages)
        self.visualizer.visualize_RSN(subject, proj.T, RSN_names)
        rsnInfo.visualize_stemplot_RSN(subject, proj, RSN_names, "proj")
        rsnInfo.visualize_stemplot_RSN(subject, mi_matrix, RSN_names, "mi")

    def get_work_folder(self):
        """
        get the path to the working folder
        :return work_folder --> path to the working folder
        """
        return self.work_folder


if __name__ == '__main__':

    if Mode == "Burbu":
        work_folder = "C:/Users/AGATHA/Desktop/4t_GEB/TFG/"
        data_loader = DataLoader(work_folder)
    else:
        work_folder = ADNI_A.base_folder
        data_loader = ADNI_A.ADNI_A()

    pipeline = Pipeline(work_folder)

    if type == "mean":
        print('Loading average HC matrix...')
        if Mode == "Burbu":
            SC_avg_HC = data_loader.get_group_avg_matrix("HC")
        else:
            SC_avg_HC = data_loader.get_AvgSC_ctrl("HC")

        print(SC_avg_HC)
        print(f"Running pipeline for average HC")
        pipeline.run_structural_connectivity(SC_avg_HC, "mean_HC")

    else:
        connectomes_path = os.path.join(pipeline.work_folder, "dades", "conncetomes", "conncetomes")
        # get the list of all subject directories
        subjects = [subject for subject in os.listdir(connectomes_path) if
                    os.path.isdir(os.path.join(connectomes_path, subject))]

        # iterate over all subjects and run the pipeline for each
        for subject in subjects:
            print('Loading matrix...')
            # loads Structural Connectivity for the subject specified
            if Mode == "Gus":
                SC = self.data_loader.get_subjectData(subject)[subject]['SC']
            else:
                SC = self.data_loader.load_matrix(subject)
            print(f"Running pipeline for subject: {subject}")
            pipeline.run_structural_connectivity(SC, subject)  # run the pipeline for each subject

    print('The whole tasks have been done!!!')
    print('The whole tasks have been done!!!')