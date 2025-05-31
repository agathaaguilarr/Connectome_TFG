# import my Classes
from computeFC import ComputeFC
from harmonicCalculator import HarmonicCalculator
from brainVisualizer import BrainVisualizer
from projecter import Projecter
from RestingStateNetworks import RestingStateNetworks
from mutualInfo import mutualInformation
from recosntructionError import reconstructionError
import p_values

# import other packages
import os
import matplotlib.pyplot as plt
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
        self.RSN_dictionary, self.RSN_names = self.rsnInfo.getRSNinformation()
        self.computeFC = ComputeFC() # creates an instance of ComputFC to obtain the FC matrixs

    def _getWorkFolder(self):
        """
        get the path to the working folder
        :return work_folder --> path to the working folder
        """
        return self.work_folder

    def _compute_e(self, matrix):
        """
        computes the eigenvectors and eigenvalues of the matrix, this matrix is normalized before the computation
        :param matrix: nxn matrix, it can be the SC matric or the FC matrix
        :return: sorted eigenvectors
        """
        # normalize SC matrix
        M = (matrix / np.max(matrix)) * 1.0
        # remove self connections (diagonal)
        M -= np.diag(np.diag(M))

        # compute the harmonics (eigenvectors)
        print('Computing harmonics...')
        e_val, e_vec = self.harmonic_calculator.compute_harmonics(M)
        return e_vec

    def _projectRSN(self, e_vec, RSN_dictionary):
        """
        Project the eigenvector e_vec with all the RSN in the RSN_dictionary
        :param e_vec: eigenvector
        :param RSN_dictionary: RSN dictionary that contains the names for each RSN and its binary (activation) vector
        :return: the projection for analysis and plots of e_vec with the RSN,
        """
        print("Projecting RSN information...")
        projecter = Projecter(e_vec)  # self.phi = e_vec!!!
        # project all the RSN
        proj = projecter.projectVectorRegion(RSN_dictionary, False)
        proj_for_plots = projecter.projectVectorRegion(RSN_dictionary, True)
        return proj, proj_for_plots

    def _projectfMRI(self, e_vec, fMRIs):
        """
        Project the eigenvector e_vec with all the fMRI in the fMRI_dictionary
        :param e_vec: eigenvector
        :param fMRIs: fMRI list that contains the fMRI for all subjects
        :return: the projection of e_vec with the fMRIS, a list of matrixs (360,T)
        """
        print("Projecting fMRI information...")
        projecter = Projecter(e_vec)  # self.phi = e_vec!!!
        # project all the fMRIs
        all_proj = []
        for i, fmri in enumerate(fMRIs):
            proj = projecter.projectVectorTime(fmri, False)  # (360, T)
            all_proj.append(proj)
        return all_proj  # a list of matrixs, each matrix is the projection of the FC/SC with an fMRI

    def _getRSNinfo(self):
        """
        obtains the Resting State Networks information from the corresponding folder
        :return: returns a dictionary for the RSN information
        """
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
        """
        it computes and plots the reconstruction error (accumulated error)
        :param recError: the instance for the reconstruction error class
        :param RSN_dictionary: the Resting State Networks dictionary
        :param RSN_names: the Resting State Networks names
        :param proj: the projections of e_vec and RSN
        :param subject: the subject id
        :return: no return
        """
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

    def _print10higherValues(self, proj, m):
        """
        Display the top 10 values for a given vector
        :param proj: vector of values
        :param m: type of values
        """
        top_n = 10
        if m == "mi":
            name = "Mutual information"
        else:
            name = "Projection"
        for row in range(proj.shape[0]):
            row_vals = proj[row, :]
            top_indices = np.argsort(row_vals)[-top_n:][::-1]

            print(f"\nRSN {self.RSN_names[row]} — Top {top_n} eigenvectors:")
            for rank, idx in enumerate(top_indices):
                val = row_vals[idx]
                print(f"  {rank + 1}. Eigenvector {idx} → {name} value: {val:.4f}")

    def _staticRun(self, M, subject):
        """
        Processes the Structural Connectivity (SC) or the Functional CConnectivity (FC) matrix for a given subject (or an average) by computing
        its harmonic components and generating 3D brain visualizations among other operations
        """
        # RSN DICTIONARY: generate the RSN dictionary
        RSN_dictionary = self.RSN_dictionary
        RSN_names = self.RSN_names

        # HARMONICS: compute the harmonics (eigenvectors)
        e_vec = self._compute_e(M)[:RSN_dictionary.shape[0], :]
        self.visualizer.visualize_RSN(subject, e_vec, list(range(0, 19))) # 1st 20s

        # PROJECT RSN (WITH THE EIGENVECTORS) AND SHOW STEM PLOTS
        proj, proj_for_plots = self._projectRSN(e_vec, RSN_dictionary)
        self._print10higherValues(proj_for_plots, "proj")

        #MUTUAL INFORMATION
        # compute the mutual information between each RSN and the eigenvectors (harmonics)
        print("Calculating Mutual Information between RSN and harmonics...")
        mi = mutualInformation(e_vec, RSN_dictionary, self.work_folder)
        mi_matrix = mi.computeMutualInformation()

        # BRAIN PLOTS FOR RSN PROJECTIONS:
        # the projection used here will be the one for plots --> better for visual comprehension
        self.visualizer.visualize_RSN(subject, proj_for_plots.T, RSN_names)  # colored brain plots for each RSN

        #STEM PLOTS FOR: PROJECTIONS (IMPORTANCE) AND MUTUAL INFORMATION
        mi.plot_stemplot(subject, proj_for_plots, RSN_names, "proj")  # projections stem plot
        mi.plot_stemplot(subject, mi_matrix, RSN_names, "mi")  # mutual information stem plot
        self._print10higherValues(mi_matrix, "mi")

        # COMPUTING AND PLOTTING RECONSTRUCTION ERRORS FOR EACH RSN
        # calculate reconstruction errors for recreating FIG.3 S.Atasoy
        recError = reconstructionError(e_vec)
        self._recErrorComputeandPlot(recError, RSN_dictionary, RSN_names, proj, subject)

        # OTHER OPERATIONS AND PLOTS
        #plot the sorted reconstruction error only for the "Default" mode
        desired_RSN, desired_proj = self.rsnInfo.get_desired_RSN(RSN_names, RSN_dictionary, proj, "Default") # select a RSN (normally the Default one)
        sorted_errors = recError.accumulated_reconstruction_error(desired_proj, desired_RSN, True) # compute the errors sorting
        recError.plot_reconstruction_error(sorted_errors, Mode, "Default", True) #plot

        return proj_for_plots

    def _dynamicRun(self, M, projRSN, rsn_name="Default", top_n=30, c=""):
        """
        Processes the Structural Connectivity (SC) or the Functional CConnectivity (FC) matrix for a given subject (or an average) by
        conducting the dynamic analysis
        :param M --> a matrix m*m (FC or SC)
        :param projRSN --> a matrix RSN
        :param rsn_name --> the name of the RSN, default by default
        :param top_n --> the number of eigenvectors to compute
        """
        if rsn_name not in self.RSN_names:
            raise ValueError(f"RSN '{rsn_name}' is not existing.")
        rsn_index = self.RSN_names.index(rsn_name)

        # obtain the eigenvectors depending on the RSN projections importance
        # we will get the most important harmonics for each RSN
        proj_values = projRSN[rsn_index] # the desired RSN projections
        sorted_indices = np.argsort(-np.abs(proj_values))  # descendent sortig by modulus (abs value)
        e_vec = self._compute_e(M) # compute the eigenvectors for the M connectivity
        e_vec_sorted = e_vec[:, sorted_indices[:top_n]]

        e = e_vec_sorted
        idx = sorted_indices

        # load the fMRIs!
        fMRI_HC = self.data_loader.get_all_fMRI()
        fMRI_MCI = self.data_loader.get_all_fMRI("MCI")
        fMRI_AD = self.data_loader.get_all_fMRI("AD")

        # project the eigenvectors computed with the control average FC/SC matrices with the fMRI
        projfMRI_HC = self._projectfMRI(e, fMRI_HC)
        projfMRI_MCI = self._projectfMRI(e, fMRI_MCI)
        projfMRI_AD = self._projectfMRI(e, fMRI_AD)

        self._plotComparisonAcrossLabels2(projfMRI_HC, projfMRI_MCI, projfMRI_AD, idx, c)

    def _runDynamicAllRSN(self, M, projRSN, c, top_n=30):
        """
        runs all the dynamic analysis the RSN for
        :param M: matrix m*m (FC or SC)
        :param projRSN: matrix RSN
        :param c: name
        :param top_n: number of eigenvectors to compute
        """
        for rsn_name in self.RSN_names:
            print(f"\nDynamic analysis for RSN: {rsn_name}")
            self._dynamicRun(M, projRSN, rsn_name=rsn_name, top_n=top_n, c=str(c) + f"_{rsn_name}")

    def _plotComparisonAcrossLabels2(self, projfMRI_HC, projfMRI_MCI, projfMRI_AD, idx, c):
        """
        plots the comparison across labels for each group for some eigenvectors
        :param projfMRI_HC: fMRI projections for HC
        :param projfMRI_MCI: fMRI projections for MCI
        :param projfMRI_AD: fMRI projections for HAD
        :param idx: order of the eigenvectors
        """
        # creates a figure with 2 rows and 5 columns
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        axs = axs.flatten()  # to loop it as a list

        # plot the barplots for the 1st 10 eigenvectors
        for i, row in enumerate(idx[:10]):
            dicctionary = self._getDictionary(i, projfMRI_HC, projfMRI_MCI, projfMRI_AD)
            print("Eigenvector " + str(row) + ": ")
            p_values.plotComparisonAcrossLabels2Ax(
                axs[i],
                tests=dicctionary,
                columnLables=["HC", "MCI", "AD"],
                graphLabel=f"EigVec {row}",
                test='Mann-Whitney',
                comparisons_correction="BH"
            )

        plt.tight_layout()
        save_dir = os.path.join(self.work_folder, "images", "barplot_comparisons_dynamic")
        os.makedirs(save_dir, exist_ok=True)  # create base directory

        # Choose subfolder based on condition
        if "FC" in c:
            save_dir_C = os.path.join(save_dir, "FC")
        else:  # "SC" in c
            save_dir_C = os.path.join(save_dir, "SC")

        os.makedirs(save_dir_C, exist_ok=True)  # create subfolder

        # Save path should point into the correct subfolder
        save_path = os.path.join(save_dir_C, f"comparison_{c}.png")
        plt.savefig(save_path)
        plt.close(fig)


    def _getDictionary(self, row, proj_hc, proj_mci, proj_ad):
        """
        Creates a dictionary of the projection matrix for each patient for each group
        """
        hc = []
        mci = []
        ad = []
        all_values = []

        # get the row "row" for each matrix on the list of projections
        for h in proj_hc: # healthy
            vals = h[row].tolist()
            hc.extend(vals)
            all_values.extend(vals)

        for m in proj_mci: # mild
            vals = m[row].tolist()
            mci.extend(vals)
            all_values.extend(vals)

        for a in proj_ad: # alzheimer
            vals = a[row].tolist()
            ad.extend(vals)
            all_values.extend(vals)

        all_values = np.array(all_values)
        # extract min and max values from the whole bunch of projections
        min_val = np.min(all_values)
        max_val = np.max(all_values)

        range_val = max_val - min_val if max_val != min_val else 1.0  # in order not to divide by zero

        hc = [(x - min_val) / range_val for x in hc]
        mci = [(x - min_val) / range_val for x in mci]
        ad = [(x - min_val) / range_val for x in ad]

        dict = {
            "HC": hc, # 197*17
            "MCI": mci, # 197*9
            "AD": ad # 197*10
        }
        return dict


    def _data_config(self, data_loader):
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
    SC_avg_HC, FC_avg_HC = pipeline._data_config(data_loader)

    print(f"Running static pipeline for average controls SC")
    rsnProjSC = pipeline._staticRun(SC_avg_HC, "mean_control_SC")
    print(f"Running static pipeline for average controls FC")
    rsnProjFC = pipeline._staticRun(FC_avg_HC, "mean_control_FC")

    # static analysis --> S.Atasoy '17 (SC) & J.Vohryzek '24 (FC)
    print('Starting the DYNAMIC analysis...')

    pipeline._runDynamicAllRSN(SC_avg_HC, rsnProjSC,"mean_control_SC", 50)
    pipeline._runDynamicAllRSN(FC_avg_HC, rsnProjFC,"mean_control_FC", 50)

    print('The whole tasks have been done!!!')
