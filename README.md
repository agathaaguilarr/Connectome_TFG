# Connectome_TFG
## **Structural and Functional Connectivity in Alzheimer’s Patients: A Harmonious Approach**

**General Description**
The main objective is to analyze how Alzheimer’s disease affects brain connectivity using harmonic decomposition methods based on previous theories of Connectome Harmonics and Functional Harmonics. The pipeline processes structural and functional connectivity matrices, extracts harmonic modes through Laplacian calculation and spectral decomposition, and performs static and dynamic analyses to identify differential patterns across clinical groups: healthy controls (HC), mild cognitive impairment (MCI), and Alzheimer’s disease patients (AD).

**Technical Aspects and Code**
- Preprocessing: Use of SC and FC connectivity matrices based on the multimodal Glasser atlas (379 ROIs).
- Laplacian Calculation and Spectral Decomposition: Extraction of eigenvalues and eigenvectors to obtain harmonic modes of the connectome.
- Static Analysis: Projection of resting-state networks (RSNs) onto harmonic modes to evaluate reconstruction accuracy and alignment using linear projection coefficients and mutual information metrics, and computing the reconstruction error.
- Dynamic Analysis: Temporal projection of harmonic modes on fMRI signals to obtain time series of harmonic activation, enabling comparison of dynamic evolution between clinical groups (different stages of Alzheimer's disease).
- Statistics: Non-parametric Mann–Whitney U tests to detect significant differences in harmonic expression between HC, MCI, and AD.
- Main Libraries: NumPy, SciPy, scikit-learn for numerical and statistical processing, matplotlib/seaborn for visualization.

**Key Code Results**
- Efficient extraction of harmonic modes with strong ability to represent RSNs.
- Implementation of reconstruction error and mutual information metrics to validate harmonic representation.
- Modular pipeline supporting both static and dynamic analysis on real ADNI data.
- Scripts for statistically comparing harmonic mode expression between clinical groups.

**Usage**
The repository includes scripts and notebooks to reproduce the full analysis from data loading to result generation. Users can adapt the pipeline for other datasets or neurological conditions. Remember to adapt your paths and directories!
