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

    def compute_e(self, matrix):
        """
        computes the eigenvectors and eigenvalues of the matrix, this matrix is normalized before the computation
        :param matrix: nxn matrix, it can be the SC matric or the FC matrix
        :return: eigenvectors and eigenvalues of the matrix
        """

        # normalize SC matrix
        M = (matrix / np.max(matrix)) * 1.0
        M -= np.diag(np.diag(M))

        # compute the harmonics (eigenvectors)
        print('Computing harmonics...')
        e_val, e_vec = self.harmonic_calculator.compute_harmonics(M)
        # we don't care about the eigen values!
        return e_vec

    def run_structural_connectivity(self, SC, subject):
        """
        Processes the Structural Connectivity (SC) matrix for a given subject by computing its harmonic components
        and generating 3D brain visualizations.
        """

        # HARMONICS: compute the harmonics (eigenvectors)
        e_vec = self.compute_e(SC)

        # RSN DICTIONARY: generate the RSN dictionary
        print("Getting the RSN dictionary")
        # obtain the path where the RSN info is stored
        if Mode == "Burbu":
            parc_path = self.work_folder + r"glasser360/"
            save_path = os.path.join(self.work_folder, "images")
        else:
            parc_path = WBF.WorkBrainProducedDataFolder + '_Parcellations/'
            save_path = './_Results/'
        rsn_path = parc_path + 'Glasser360RSN_7_RSN_labels.csv'
        # get the information
        rsnInfo = RestingStateNetworks(rsn_path, save_path)
        RSN_dictionary, RSN_names = rsnInfo.getRSNinformation()

        # PROJECT RSN (WITH THE EIGENVECTORS)
        print("Projecting RSN information...")
        e_vec = e_vec[:RSN_dictionary.shape[0], :] # same regions in both e_vec and dictionary
        projecter = Projecter(e_vec)  # self.phi = e_vec!!!
        # project all the RSN
        proj = projecter.projectVectorRegion(RSN_dictionary, False)
        proj_for_plots = projecter.projectVectorRegion(RSN_dictionary, True)

        #MUTUAL INFORMATION
        # compute the mutual information between each RSN and the eigenvectors (harmonics)
        print("Calculating Mutual Information between RSN and harmonics...")
        mi_matrix = projecter.computeMutualInformation(RSN_dictionary)

        # BRAIN PLOTS FOR RSN PROJECTIONS:
        # the projection used here will be the one for plots --> better for visual comprehension
        self.visualizer.visualize_RSN(subject, proj_for_plots.T, RSN_names)  # colored brain plots for each RSN

        #STEM PLOTS FOR: PROJECTIONS (IMPORTANCE) AND MUTUAL INFORMATION
        rsnInfo.visualize_stemplot_RSN(subject, proj_for_plots, RSN_names, "proj")  # projections stem plot
        rsnInfo.visualize_stemplot_RSN(subject, mi_matrix, RSN_names, "mi")  # mutual information stem plot

        # COMPUTING AND PLOTTING RECONSTRUCTION ERRORS FOR EACH RSN
        # calculate reconstruction errors for recreating FIG.3 S.Atasoy
        errors_dict = {}
        for rsn in RSN_names:
            # obtain the desired RSN and its projection
            desired_RSN, desired_proj = rsnInfo.get_desired_RSN(RSN_names, RSN_dictionary, proj, rsn) # no sorted
            # compute the reconstruction error
            errors = projecter.accumulated_reconstruction_error(desired_proj, desired_RSN)
            # save it in the errors dictionary
            errors_dict[rsn] = errors
        # generate the plot
        rsnInfo.plot_all_reconstruction_errors(errors_dict, Mode)

        # OTHER OPERATIONS AND PLOTS
        #plot the sorted reconstruction error only for the "Default" mode
        desired_RSN, desired_proj = rsnInfo.get_desired_RSN(RSN_names, RSN_dictionary, proj, "Default") # select a RSN (normally the Default one)
        sorted_errors = projecter.accumulated_reconstruction_error(desired_proj, desired_RSN, True) # compute the errors sorting
        rsnInfo.plot_reconstruction_error(sorted_errors, Mode, rsn, True) #plot

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
            SC_avg_HC = data_loader.get_group_avg_matrix("HC") # + sc ??
            #FC_avg_HC = data_loader.get_group_avg_matrix("HC", fmri)
        else:
            SC_avg_HC = data_loader.get_AvgSC_ctrl("HC")

        print(f"Running pipeline for average HC")
        pipeline.run_structural_connectivity(SC_avg_HC, "mean_HC")

    else: #Gus Mode
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