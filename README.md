# Parkinson's Disease Classification from T1w MRI using FSL and Scikit-learn

## Project Overview

This project implements a pipeline to preprocess T1-weighted (T1w) structural MRI scans and train machine learning models to classify participants as having Parkinson's Disease (PD) or as Healthy Controls (HC). The primary data source is the Parkinson's Progression Markers Initiative (PPMI).

The methodology focuses on using a standard FSL-based preprocessing workflow combined with atlas-based Region of Interest (ROI) feature extraction and common Scikit-learn classifiers, aiming for a balance between standard neuroimaging practices and accessibility.

## Prerequisites

### System Dependencies:
*   **Operating System:** Linux environment (tested on Windows Subsystem for Linux - WSL).
*   **FSL (v6.0.x recommended):** FMRIB Software Library. Ensure the following tools are installed and accessible from the command line:
    *   `fslreorient2std`
    *   `bet` (Brain Extraction Tool)
    *   `flirt` (Linear Registration)
    *   `fnirt` (Non-linear Registration)
    *   `applywarp`
    *   `fslstats`
    *   `fslmaths`
    *   **Important:** The FSL environment variables (`FSLDIR`, `PATH`) must be correctly configured in your shell (e.g., via `.bashrc`) AND within the Python environment (see script headers).
*   **dcm2niix:** For converting DICOM images to NIfTI format. Ensure it's installed and in your system's PATH.
*   **bc:** Required by FSL's `bet` script for the `dc` (desk calculator) command. Install via `sudo apt-get install bc` on Debian/Ubuntu-based systems.

### Python Environment:
*   Python 3.8+ (tested with 3.9)
*   Required libraries: Create an environment (e.g., using Conda) and install using pip:
    ```bash
    pip install pandas numpy scipy scikit-learn nilearn nibabel joblib matplotlib seaborn
    ```
    *(Or use a `requirements.txt` file if provided)*



## Script Descriptions

1.  **`1dicom_conversion.py`**: Converts DICOM series to NIfTI using `dcm2niix`.
2.  **`2orientation_standard.py`**: Reorients NIfTI files to FSL standard orientation using `fslreorient2std`.
3.  **`3brainextraction.py`**: Performs brain extraction using FSL `bet` (parallelized). **Requires parameter tuning and QC.**
4.  **`4mni.py`**: Performs linear (`flirt`) and non-linear (`fnirt`, `applywarp`) registration to MNI space (parallelized). **Requires QC.**
5.  **`5intensity_normalization.py`**: Performs Z-score intensity normalization within the brain mask using FSL tools (parallelized).
6.  **`6feature_engineering.py`**: Extracts mean, std dev, skewness, and kurtosis from specified Harvard-Oxford subcortical ROIs using Nilearn.
7.  **`7machinelearning.py`**: Implements StandardScaler->PCA->Classifier pipelines, tunes hyperparameters using GridSearchCV, evaluates using cross-validation, and saves results/plots.


## Notes

*   **Brain Extraction Tuning:** Careful tuning of the `bet_f_threshold` and potentially the `-B` option in `3brainextraction.py` is crucial for good results. Visual QC is mandatory.
*   **FSL Environment in Python:** The scripts include code to set FSL environment variables (`FSLDIR`, `PATH`) within the Python process. Ensure the `fsl_dir` path is correct in those scripts.
*   **Performance:** Registration (`4mni.py`) is computationally intensive. Parallelization helps but may still take significant time depending on the number of subjects and hardware. Running on standard cloud instances like Codespaces can be very slow for these tasks.
*   **Metadata Management:** If reprocessing failed subjects, carefully manage the different metadata CSV files generated at each step to track which files belong to which batch and ensure the correct inputs are used for subsequent steps before potentially merging the final successful metadata.

---
