# import my Classes
from computeFC import ComputeFC
from harmonicCalculator import HarmonicCalculator
from brainVisualizer import BrainVisualizer
from projecter import Projecter
from RestingStateNetworks import RestingStateNetworks
from mutualInfo import mutualInformation
from recosntructionError import reconstructionError
import os
import pandas as pd
import matplotlib.pyplot as plt

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
        else: # Mode == Burbu
            self.data_loader = DataLoader(self.work_folder) # initializes the DataLoader, to load the information from the work_folder

        self.harmonic_calculator = HarmonicCalculator(th=0.00065) # initialize an instance of the HarmonicCalculator class
        self.visualizer = BrainVisualizer(self.work_folder, Mode) # creates an instance of BrainVisualizer to generate plots
        self.rsnInfo = self._getRSNinfo() # creates an instance of the RSNinfo, to obtain the information for the Resting State Networks
        self.computeFC = ComputeFC() # creates an instance of ComputFC to obtain the FC matrixs

    def _compute_e(self, matrix):
        """
        computes the eigenvectors and eigenvalues of the matrix, this matrix is normalized before the computation
        :param matrix: nxn matrix, it can be the SC matric or the FC matrix
        :return: sorted eigenvectors
        """
        # normalize SC matrix
        #M = matrix
        M = (matrix / np.max(matrix)) * 1.0
        # remove self connections (diagonal)
        M -= np.diag(np.diag(M))

        # compute the harmonics (eigenvectors)
        print('Computing harmonics...')
        e_val, e_vec = self.harmonic_calculator.compute_harmonics(M)
        return e_vec

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

    def _staticRun(self, SC, subject):
        """
        Processes the Structural Connectivity (SC) or the Functional CConnectivity (FC) matrix for a given subject (or an average) by computing
        its harmonic components and generating 3D brain visualizations among other operations
        """
        # RSN DICTIONARY: generate the RSN dictionary
        RSN_dictionary, RSN_names = self.rsnInfo.getRSNinformation()

        # HARMONICS: compute the harmonics (eigenvectors)
        e_vec = self._compute_e(SC)[:RSN_dictionary.shape[0], :]

        # PROJECT RSN (WITH THE EIGENVECTORS) AND SHOW STEM PLOTS
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
        mi.plot_stemplot(subject, proj, RSN_names, "proj")  # projections stem plot
        mi.plot_stemplot(subject, mi_matrix, RSN_names, "mi")  # mutual information stem plot

        # COMPUTING AND PLOTTING RECONSTRUCTION ERRORS FOR EACH RSN
        # calculate reconstruction errors for recreating FIG.3 S.Atasoy
        recError = reconstructionError(e_vec)
        self._recErrorComputeandPlot(recError, RSN_dictionary, RSN_names, proj, subject)

        # OTHER OPERATIONS AND PLOTS
        #plot the sorted reconstruction error only for the "Default" mode
        desired_RSN, desired_proj = self.rsnInfo.get_desired_RSN(RSN_names, RSN_dictionary, proj, "Default") # select a RSN (normally the Default one)
        sorted_errors = recError.accumulated_reconstruction_error(desired_proj, desired_RSN, True) # compute the errors sorting
        recError.plot_reconstruction_error(sorted_errors, Mode, "Default", True) #plot

    def computeMeanProjections(self, projTimeSC, projTimeFC):
        """
        computes the mean projection (e_Vec projected to fMRI) matrix
        :param projTimeSC: all the SC projections
        :param projTimeFC: all the FC projections
        :return: the mean projection (rows: the mean for the projections, columns: time steps)
        """
        projSC_array = np.stack(projTimeSC)  # (N, k, T)
        projFC_array = np.stack(projTimeFC)  # (N, k, T)

        # mean of all the vectors for each time step
        meanProjSC = projSC_array.mean(axis=0)  # (k, T)
        meanProjFC = projFC_array.mean(axis=0)  # (k, T)

        return meanProjSC, meanProjFC

    def _dynamicRun(self, SCs, FCs , fMRIs, group):
        """
         Processes the Structural Connectivity (SC) or the Functional CConnectivity (FC) matrix for a given subject (or an average) by computing
        its harmonic components and projecting them with the corresponding fMRI
        :param SCs: a list with all the SC matrix for a condition
        :param FCs: a list with all the FC matrix for a condition
        :param fMRIs: a list with all the fMRI matrix for a condition
        """
        # structural connectivity
        projTimeSC = []
        for SC, fMRI in zip(SCs, fMRIs):
            e_vec = self._compute_e(SC)
            projecter = Projecter(e_vec)
            projTime = projecter.projectVectorTime(fMRI)
            projTimeSC.append(projTime)
        projTimeFC = []

        # functional connectivity
        for FC, fMRI in zip(FCs, fMRIs):
            e_vec = self._compute_e(FC)
            projecter = Projecter(e_vec)
            projTime = projecter.projectVectorTime(fMRI)
            projTimeFC.append(projTime)

        # mean
        meanProjTimeSC, meanProjTimeFC = self.computeMeanProjections(projTimeSC, projTimeFC)

        



    def _geTWorkFolder(self):
        """
        get the path to the working folder
        :return work_folder --> path to the working folder
        """
        return self.work_folder

    def _data_config_static(self, data_loader):
        """
        computes the mean (average) for all the subjects in the HC (control condition) / is possible to change the condition!!!
        @param data_loader --> the data loader
        @:return SC_avg_group, FC_avg_group a matrix corresponding to the mean (of all subject's) the SC/FC for the indicated condition
        """
        condition = "HC"
        if Mode == "Burbu": # Burbu Mode
            # compute the mean for all SC matrixs
            SC_avg_group = data_loader.get_group_avg_matrix(condition, sc=True)
            # compute the mean for all FC matrixs
            FC_avg_group = data_loader.get_group_avg_matrix(condition, sc=False)
        else:  # Gus Mode
            SC_avg_group = data_loader.get_AvgSC_ctrl(condition)
            data = data_loader.get_fullGroup_data(condition)
            FCs = [np.corrcoef(data[s]["timeseries"][:360],rowvar=True) for s in data]
            FC_avg_group = np.mean(FCs, axis=0)

        return SC_avg_group, FC_avg_group


    def _loadSubjectAllSubjects(self, data_loader, group_name="HC"):
        """
        loads SC, FC and fMRI for each subject on a clinic condition
        :param data_loader: DataLoader
        :param group_name: condition ("HC", "MCI", "AD")
        :return: lists with all SCs, FCs, fMRIs, subject_ids for each subject
        """
        print(f"Loading SC, FC, and fMRI data for group: {group_name}")

        # obtain all the subjects for an specific condition
        group_subjects = data_loader.get_classification()
        subjects_in_group = [subject for subject, condition in group_subjects.items() if condition == group_name]

        # prepare empty lists
        SCs = []
        FCs = []
        fMRIs = []
        subject_ids = []

        # loop all the subjects for the specified conditon
        for subject_id in subjects_in_group:
            # get the SC matrix for the subject
            sc_matrix = data_loader.load_matrix(subject_id, sc=True)
            if sc_matrix is not None: # make sure is not Null
                SCs.append(sc_matrix) # add it to the SC list

            # get the FC matrix (& compute it) and the fMRI for the subject
            raw_fmri_data = data_loader.load_matrix(subject_id, sc=False)
            if raw_fmri_data is not None: # make sure is not Null
                fc_matrix = self.computeFC.compute_from_fmri(raw_fmri_data) # compute the FC from the raw fMRI
                # add it to the FC and fMRI lists, respectively
                FCs.append(fc_matrix)
                fMRIs.append(raw_fmri_data)

            # add the subject ID to the ids list
            subject_ids.append(subject_id)

        return SCs, FCs, fMRIs, subject_ids


if __name__ == '__main__':

    if Mode == "Burbu":
        work_folder = "C:/Users/AGATHA/Desktop/4t_GEB/TFG/"
        data_loader = DataLoader(work_folder)
    else:
        work_folder = ADNI_A.base_folder
        data_loader = ADNI_A.ADNI_A()

    pipeline = Pipeline(work_folder)

    # static analysis --> S.Atasoy '16 (SC) & K.Glomb '21 (FC)
    print('Starting the STATIC analysis...')
    print('Loading average matrixs...')
    SC_avg_HC, FC_avg_HC = pipeline._data_config_static(data_loader)

    print(f"Running static pipeline for average controls SC")
    pipeline._staticRun(SC_avg_HC, "mean_control_SC")
    print(f"Running static pipeline for average controls FC")
    pipeline._staticRun(FC_avg_HC, "mean_control_FC")

    # static analysis --> S.Atasoy '17 (SC) & J.Vohryzek '24 (FC)
    print('Starting the DYNAMIC analysis...')
    SCs_HC, FCs_HC, fMRIs_HC, subject_ids_HC = pipeline._loadSubjectAllSubjects(data_loader)
    SCs_MCI, FCs_MCI, fMRIs_MCI, subject_ids_MCI = pipeline._loadSubjectAllSubjects(data_loader, "MCI")
    SCs_AD, FCs_AD, fMRIs_AD, subject_ids_AD = pipeline._loadSubjectAllSubjects(data_loader, "AD")

    pipeline._dynamicRun(SCs_HC, FCs_HC, fMRIs_HC, "HC")
    pipeline._dynamicRun(SCs_MCI, FCs_MCI, fMRIs_MCI, "MCI")
    pipeline._dynamicRun(SCs_AD, FCs_AD, fMRIs_AD, "AD")


    print('The whole tasks have been done!!!')