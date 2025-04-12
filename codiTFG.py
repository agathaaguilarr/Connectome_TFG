# import my Classes
from harmonicCalculator import HarmonicCalculator
from brainVisualizer import BrainVisualizer
from projecter import Projecter
from RestingStateNetworks import RestingStateNetworks
from mutualInfo import mutualInformation
from recosntructionError import reconstructionError
import os
import pandas as pd

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
            self.data_loadera = ADNI_A.ADNI_A()
        else:
            self.data_loader = DataLoader(self.work_folder) # initializes the DataLoader, to load the information from the work_folder
        self.harmonic_calculator = HarmonicCalculator(th=0.00065) # initialize an instance of the HarmonicCalculator class
        self.visualizer = BrainVisualizer(self.work_folder, Mode) # creates an instance of BrainVisualizer to generate plots
        self.rsnInfo = self._getRSNinfo()

    def _compute_e(self, matrix):
        """
        computes the eigenvectors and eigenvalues of the matrix, this matrix is normalized before the computation
        :param matrix: nxn matrix, it can be the SC matric or the FC matrix
        :return: sorted eigenvectors
        """

        # normalize SC matrix
        M = (matrix / np.max(matrix)) * 1.0
        M -= np.diag(np.diag(M))
        df = pd.DataFrame(matrix)
        print(df)
        # compute the harmonics (eigenvectors)
        print('Computing harmonics...')
        e_val, e_vec = self.harmonic_calculator.compute_harmonics(M)

        idx = np.argsort(e_val)[::-1]  # sort from greater to smaller
        eigenvectors = e_vec[:, idx]

        return eigenvectors

    def _project(self, e_vec, RSN_dictionary):
        print("Projecting RSN information...")
        projecter = Projecter(e_vec)  # self.phi = e_vec!!!
        # project all the RSN
        proj = projecter.projectVectorRegion(RSN_dictionary, False)
        proj_for_plots = projecter.projectVectorRegion(RSN_dictionary, True)
        return proj, proj_for_plots

    def _getRSNinfo(self):
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
        return rsnInfo

    def _recErrorComputeandPlot(self, recError, RSN_dictionary, RSN_names, proj, subject):
        errors_dict = {}
        for rsn in RSN_names:
            # obtain the desired RSN and its projection
            desired_RSN, desired_proj = self.rsnInfo.get_desired_RSN(RSN_names, RSN_dictionary, proj, rsn)  # no sorted
            # compute the reconstruction error
            errors = recError.accumulated_reconstruction_error(desired_proj, desired_RSN)
            # save it in the errors dictionary
            errors_dict[rsn] = errors
        # generate the plot
        recError.plot_all_reconstruction_errors(errors_dict, Mode, subject)

    def run(self, SC, subject):
        """
        Processes the Structural Connectivity (SC) matrix for a given subject by computing its harmonic components
        and generating 3D brain visualizations.
        """
        # RSN DICTIONARY: generate the RSN dictionary
        RSN_dictionary, RSN_names = self.rsnInfo.getRSNinformation()

        # HARMONICS: compute the harmonics (eigenvectors)
        e_vec = self._compute_e(SC)[:RSN_dictionary.shape[0], :]

        # PROJECT RSN (WITH THE EIGENVECTORS)
        proj, proj_for_plots = self._project(e_vec, RSN_dictionary)

        #MUTUAL INFORMATION
        # compute the mutual information between each RSN and the eigenvectors (harmonics)
        print("Calculating Mutual Information between RSN and harmonics...")
        mi = mutualInformation(e_vec, RSN_dictionary, self.work_folder)
        mi_matrix = mi.computeMutualInformation()

        # BRAIN PLOTS FOR RSN PROJECTIONS:
        # the projection used here will be the one for plots --> better for visual comprehension
        self.visualizer.visualize_RSN(subject, proj_for_plots.T, RSN_names)  # colored brain plots for each RSN

        #STEM PLOTS FOR: PROJECTIONS (IMPORTANCE) AND MUTUAL INFORMATION
        self.rsnInfo.visualize_stemplot_RSN(subject, proj, RSN_names, "proj")  # projections stem plot
        self.rsnInfo.visualize_stemplot_RSN(subject, mi_matrix, RSN_names, "mi")  # mutual information stem plot

        # COMPUTING AND PLOTTING RECONSTRUCTION ERRORS FOR EACH RSN
        # calculate reconstruction errors for recreating FIG.3 S.Atasoy
        recError = reconstructionError(e_vec)
        self._recErrorComputeandPlot(recError, RSN_dictionary, RSN_names, proj, subject)

        # OTHER OPERATIONS AND PLOTS
        #plot the sorted reconstruction error only for the "Default" mode
        desired_RSN, desired_proj = self.rsnInfo.get_desired_RSN(RSN_names, RSN_dictionary, proj, "Default") # select a RSN (normally the Default one)
        sorted_errors = recError.accumulated_reconstruction_error(desired_proj, desired_RSN, True) # compute the errors sorting
        recError.plot_reconstruction_error(sorted_errors, Mode, "Default", True) #plot

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
        print('Loading average matrixs...')
        if Mode == "Burbu": # Burbu Mode
            # compute the mean for all SC matrixs
            SC_avg_HC = data_loader.get_group_avg_matrix("HC")
            # compute the mean for all FC matrixs
            FC_avg_HC = data_loader.get_group_avg_matrix("HC", False)
        else:  # Gus Mode
            SC_avg_HC = data_loader.get_AvgSC_ctrl("HC")
            data = data_loader.get_fullGroup_data("HC")
            FCs = [np.corrcoef(data[s]["timeseries"][:360],rowvar=True) for s in data]
            FC_avg_HC = np.mean(FCs, axis=0)

        print(f"Running pipeline for average controls SC")
        pipeline.run(SC_avg_HC, "mean_control_SC")
        print(f"Running pipeline for average controls FC")
        pipeline.run(FC_avg_HC, "mean_control_FC")

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