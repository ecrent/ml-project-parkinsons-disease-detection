# --- Feature Engineering Script (using Nilearn and Harvard-Oxford Atlas) ---

import pandas as pd
import os
import numpy as np
from scipy.stats import skew, kurtosis
from nilearn import image, datasets
from nilearn.maskers import NiftiMasker
import time
import warnings

# --- Filter RuntimeWarnings from scipy.stats (often due to constant data in small ROIs) ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module='scipy.stats._moment')

# --- 1. Configuration ---
print("\n--- Configuration for Feature Engineering ---")
project_dir = '/home/ecren/pd_project'

# --- Input from previous step ---
input_metadata_path = os.path.join(project_dir, 'normalized_metadata.csv')
# Input NIfTI directory is implicitly defined by paths in the metadata

# --- Output for this step ---
output_features_dir = os.path.join(project_dir, 'features') # Directory to save features
output_features_csv = os.path.join(output_features_dir, 'extracted_features_ho_thr25.csv') # The feature matrix

# --- Atlas Configuration ---
# Use thr25 for more specific ROIs, 1mm resolution
atlas_name = 'harvard_oxford'
atlas_variant = 'cort-maxprob-thr25-1mm' # Cortical part
atlas_sub_variant = 'sub-maxprob-thr25-1mm' # Subcortical part
# -----------------------------------------
print(f"INFO: Input metadata --> {input_metadata_path}")
print(f"INFO: Output features file --> {output_features_csv}")
print(f"INFO: Using Harvard-Oxford Atlas (Cort: {atlas_variant}, Sub: {atlas_sub_variant})")

# --- Create necessary output directory ---
os.makedirs(output_features_dir, exist_ok=True)

# --- 2. Load the Metadata from the Normalization Step ---
print(f"\n--- Loading Normalized Metadata: {input_metadata_path} ---")
try:
    normalized_df = pd.read_csv(input_metadata_path)
    print(f"Loaded metadata for {len(normalized_df)} subjects.")
    required_cols = ['normalized_mni_path', 'Subject', 'Group'] # Need path, ID, and label
    missing_cols = [col for col in required_cols if col not in normalized_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from metadata: {', '.join(missing_cols)}")

    if normalized_df['normalized_mni_path'].isnull().any():
        print("WARNING: Metadata contains rows with missing 'normalized_mni_path'. Filtering these out.")
        initial_count = len(normalized_df)
        normalized_df = normalized_df.dropna(subset=['normalized_mni_path']).copy()
        print(f"Processing {len(normalized_df)} subjects after removing {initial_count - len(normalized_df)} rows.")

except FileNotFoundError:
    print(f"ERROR: Metadata file not found at {input_metadata_path}")
    raise SystemExit("Exiting. Did the previous normalization step run correctly?")
except Exception as e:
    print(f"ERROR loading normalized metadata: {e}")
    raise SystemExit("Exiting.")

if normalized_df.empty:
    raise SystemExit("ERROR: Normalized metadata is empty or all entries have missing paths.")

# Get list of NIfTI file paths to process
nifti_files = normalized_df['normalized_mni_path'].tolist()
subject_ids = normalized_df['Subject'].tolist()
group_labels = normalized_df['Group'].tolist()


# --- 3. Fetch and Prepare Harvard-Oxford Atlas ---
print("\n--- Fetching and Preparing Harvard-Oxford Atlas ---")
try:
    print(f"Fetching cortical atlas: {atlas_variant}")
    atlas_cort = datasets.fetch_atlas_harvard_oxford(atlas_variant)
    atlas_cort_img = image.load_img(atlas_cort.maps)
    atlas_cort_labels = ['HO_Cort_' + label.replace(' ', '_').replace('-', '_').replace(',', '') for label in atlas_cort.labels] # Prepend type, clean names
    print(f"  Cortical Labels: {len(atlas_cort_labels) - 1}") # -1 for background

    print(f"Fetching subcortical atlas: {atlas_sub_variant}")
    atlas_sub = datasets.fetch_atlas_harvard_oxford(atlas_sub_variant)
    atlas_sub_img = image.load_img(atlas_sub.maps)
    atlas_sub_labels = ['HO_Sub_' + label.replace(' ', '_').replace('-', '_').replace(',', '') for label in atlas_sub.labels] # Prepend type, clean names
    print(f"  Subcortical Labels: {len(atlas_sub_labels) - 1}")

    # --- Combine Atlases ---
    print("Combining cortical and subcortical atlases...")
    # Find max label in cortical atlas (labels usually start from 1)
    max_cort_label = int(np.max(atlas_cort_img.get_fdata()))
    print(f"  Max cortical label index: {max_cort_label}")

    # Create modified subcortical data: add max_cort_label to non-zero labels
    sub_data = atlas_sub_img.get_fdata()
    sub_modified_data = np.where(sub_data > 0, sub_data + max_cort_label, 0)
    # Create a new Nifti image for the modified subcortical atlas
    sub_modified_img = image.new_img_like(atlas_sub_img, sub_modified_data)

    # Combine cortical and modified subcortical using nilearn.image.math_img
    # Ensure background (0) remains background
    combined_atlas_img = image.math_img("img1 + img2", img1=atlas_cort_img, img2=sub_modified_img)

    # Combine labels (skip background 'BG' label at index 0 if present)
    all_labels_raw = atlas_cort.labels + atlas_sub.labels
    combined_labels_cleaned = atlas_cort_labels + atlas_sub_labels
    if combined_labels_cleaned[0].endswith("Background"):
         combined_labels_cleaned = combined_labels_cleaned[1:] # Remove background if it's the first element

    # Get unique label indices present in the combined atlas data (excluding 0)
    unique_labels = np.unique(combined_atlas_img.get_fdata())
    unique_labels = unique_labels[unique_labels > 0].astype(int)
    print(f"  Total unique ROI labels in combined atlas (excluding background): {len(unique_labels)}")
    # Ensure the number of labels matches the number of unique indices found
    if len(unique_labels) != len(combined_labels_cleaned):
         print(f"WARNING: Mismatch between number of unique label values ({len(unique_labels)}) and label names ({len(combined_labels_cleaned)}). Check atlas loading.")
         # Attempt to map found unique labels to the cleaned names list (adjusting for 1-based indexing)
         try:
             final_label_names = [combined_labels_cleaned[idx - 1] for idx in unique_labels]
             print("   Adjusted final label names based on unique indices found.")
         except IndexError:
             print("   ERROR: Could not map unique indices to label names. Aborting.")
             raise SystemExit
    else:
         final_label_names = combined_labels_cleaned

    print(f"Prepared combined atlas with {len(final_label_names)} ROIs.")

except Exception as e:
    print(f"ERROR loading or preparing atlas: {e}")
    raise SystemExit("Exiting.")


# --- 4. Feature Extraction Loop ---
print(f"\n--- Starting Feature Extraction for {len(nifti_files)} subjects across {len(unique_labels)} ROIs ---")
# Initialize a dictionary to store features: {feature_name: [list_of_values_for_all_subjects]}
features_dict = {}
# Initialize column names list
feature_columns = []

start_loop_time = time.time()

# Define statistics to compute
stats_funcs = {
    'mean': np.mean,
    'std': np.std,
    'skew': skew,
    'kurt': kurtosis
    # Add more functions here if needed (e.g., median, min, max, entropy)
}
stat_names = list(stats_funcs.keys())

# Pre-generate column names
for roi_idx, roi_name in zip(unique_labels, final_label_names):
    for stat_name in stat_names:
        col_name = f"{roi_name}_{stat_name}"
        feature_columns.append(col_name)
        features_dict[col_name] = [] # Initialize empty list for this feature

# Loop through each ROI defined by a unique label index
for i, (roi_label_index, roi_name) in enumerate(zip(unique_labels, final_label_names)):
    roi_start_time = time.time()
    print(f"Processing ROI {i+1}/{len(unique_labels)}: {roi_name} (Index: {roi_label_index})...")

    try:
        # --- Create a binary mask for the current ROI ---
        roi_binary_mask_img = image.math_img(f"img == {roi_label_index}", img=combined_atlas_img)

        # --- Check if the mask is empty ---
        # Summing the mask data tells us if any voxels belong to this ROI
        if np.sum(roi_binary_mask_img.get_fdata()) == 0:
            print(f"  WARNING: ROI {roi_name} (Index: {roi_label_index}) is empty in the atlas. Skipping.")
            # Add NaNs for all subjects for the stats of this ROI
            for stat_name in stat_names:
                col_name = f"{roi_name}_{stat_name}"
                features_dict[col_name].extend([np.nan] * len(nifti_files))
            continue # Skip to the next ROI

        # --- Initialize and fit the NiftiMasker for this ROI ---
        # Use memory caching for efficiency if processing the same files multiple times
        masker = NiftiMasker(
            mask_img=roi_binary_mask_img,
            # standardize=False, # Already done
            memory='nilearn_cache', # Cache intermediate results
            memory_level=1, # Cache level
            verbose=0 # Reduce verbosity inside the loop
        )

        # Extract data for ALL subjects within this single ROI mask
        # fit_transform returns a list of 1D arrays (one per subject)
        # containing the voxel values within the mask for that subject.
        roi_data_all_subjects = masker.fit_transform(nifti_files)

        # --- Calculate statistics for each subject for this ROI ---
        for subject_idx, subject_roi_data in enumerate(roi_data_all_subjects):
            if subject_roi_data is None or subject_roi_data.size == 0:
                 # This might happen if the subject's image doesn't overlap with the mask
                 # print(f"  Warning: No data extracted for Subject {subject_ids[subject_idx]} in ROI {roi_name}")
                 # Append NaN for all stats for this subject in this ROI
                 for stat_name in stat_names:
                      col_name = f"{roi_name}_{stat_name}"
                      features_dict[col_name].append(np.nan)
                 continue

            # Calculate each statistic
            for stat_name, func in stats_funcs.items():
                col_name = f"{roi_name}_{stat_name}"
                try:
                    # Handle potential issues like constant data for std, skew, kurt
                    if stat_name in ['std', 'skew', 'kurt'] and np.all(subject_roi_data == subject_roi_data[0]):
                         stat_value = 0.0 # Or np.nan, depending on desired handling
                    elif stat_name == 'kurt':
                         # Scipy's kurtosis is Fisher's (excess), add 3 for Pearson's if needed
                         # Default is Fisher's, which is fine for feature comparison.
                         stat_value = func(subject_roi_data, fisher=True, bias=False) # Use fisher, unbiased
                    else:
                         stat_value = func(subject_roi_data)

                    # Check for NaN or Inf result (can happen with kurtosis/skew on small/weird data)
                    if not np.isfinite(stat_value):
                        # print(f"  Warning: Non-finite result ({stat_value}) for {stat_name} in ROI {roi_name}, Subj {subject_ids[subject_idx]}. Setting to 0.")
                        stat_value = 0.0 # Or np.nan

                except Exception as stat_e:
                    print(f"  ERROR calculating {stat_name} for ROI {roi_name}, Subject {subject_ids[subject_idx]}: {stat_e}")
                    stat_value = np.nan # Use NaN for errors

                # Append the calculated value to the correct list in the dictionary
                # Ensure the list exists (should be initialized above)
                if col_name in features_dict:
                     # Check if list length matches current subject index before appending
                     if len(features_dict[col_name]) == subject_idx:
                          features_dict[col_name].append(stat_value)
                     else:
                          # This indicates a logic error or previous failure for this subject
                          print(f"  CRITICAL ERROR: Length mismatch for feature {col_name}, subject index {subject_idx}. Expected length {subject_idx}, got {len(features_dict[col_name])}.")
                          # Handle error - potentially fill with NaNs or stop
                          # For now, append NaN and print warning
                          features_dict[col_name].append(np.nan)

                else:
                     print(f"  CRITICAL ERROR: Feature column {col_name} not pre-initialized.")


        roi_end_time = time.time()
        # print(f"  Finished ROI {roi_name} in {roi_end_time - roi_start_time:.2f} seconds.")

    except Exception as roi_e:
        print(f"  ERROR processing ROI {roi_name} (Index: {roi_label_index}): {roi_e}")
        # If an error occurs for an ROI, fill its features with NaN for all subjects
        for stat_name in stat_names:
             col_name = f"{roi_name}_{stat_name}"
             # Ensure list length matches subject count if error occurred mid-subject processing
             current_len = len(features_dict.get(col_name, []))
             nans_to_add = len(nifti_files) - current_len
             if nans_to_add > 0:
                  features_dict[col_name].extend([np.nan] * nans_to_add)


# --- 5. Assemble Final DataFrame ---
print("\n--- Assembling Final Feature DataFrame ---")
try:
    # Check if all feature lists have the correct length
    expected_length = len(nifti_files)
    length_ok = True
    for col, values in features_dict.items():
        if len(values) != expected_length:
            print(f"ERROR: Feature '{col}' has incorrect length {len(values)}, expected {expected_length}.")
            length_ok = False
            # Attempt to fix by padding with NaN (use cautiously)
            # diff = expected_length - len(values)
            # if diff > 0: features_dict[col].extend([np.nan] * diff)

    if not length_ok:
         print("ERROR: Length mismatch in feature lists. Cannot reliably create DataFrame.")
         raise SystemExit("Exiting due to feature length errors.")


    # Create DataFrame from the dictionary
    features_df = pd.DataFrame(features_dict)

    # Add Subject ID and Group Label columns
    features_df['Subject'] = subject_ids
    features_df['Group'] = group_labels

    # Reorder columns to have Subject and Group first
    cols = ['Subject', 'Group'] + feature_columns
    features_df = features_df[cols]

    print(f"Feature DataFrame created with shape: {features_df.shape}")

    # --- Handle Potential NaN values ---
    nan_counts = features_df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print("\nWARNING: Features contain NaN values:")
        print(nan_cols)
        # Consider imputation strategies here if needed (e.g., fill with mean/median)
        # For now, we just report them. ML steps might need imputation.
        # Example: features_df.fillna(features_df.mean(), inplace=True) # Mean imputation
        print("  NaN values may need imputation before ML training.")

except Exception as df_e:
    print(f"ERROR creating final DataFrame: {df_e}")
    raise SystemExit("Exiting.")


# --- 6. Save Features ---
print(f"\n--- Saving Features to: {output_features_csv} ---")
try:
    features_df.to_csv(output_features_csv, index=False)
    print("Features saved successfully.")
except Exception as save_e:
    print(f"ERROR saving features CSV: {save_e}")


# --- 7. Cleanup Nilearn Cache (Optional) ---
# from nilearn.maskers import clean_maskers
# print("\n--- Cleaning Nilearn Masker Cache ---")
# clean_maskers(memory='nilearn_cache') # Clean up cache if needed


end_loop_time = time.time()
print(f"\n--- Feature Engineering Script Complete ---")
print(f"Total feature extraction time: {(end_loop_time - start_loop_time) / 60:.2f} minutes.")
print(f"NEXT STEP: Machine Learning using the features in '{output_features_csv}'")