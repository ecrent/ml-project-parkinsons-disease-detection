# --- Orientation Standardization Script (using fslreorient2std) ---

import os
import sys
import subprocess

print("--- Setting FSL Environment Variables for Python ---")

# --- IMPORTANT: Verify this path matches your FSL installation ---
fsl_dir = "/workspaces/codespaces-jupyter/fsl"
# -------------------------------------------------------------

fsl_bin_path = os.path.join(fsl_dir, 'bin')

# Check if FSLDIR is set and correct
current_fsldir = os.environ.get('FSLDIR')
if current_fsldir != fsl_dir:
    print(f"Setting FSLDIR environment variable to: {fsl_dir}")
    os.environ['FSLDIR'] = fsl_dir
else:
    print(f"FSLDIR already set correctly: {current_fsldir}")

# Check if FSL bin directory is in PATH
current_path = os.environ.get('PATH', '')
if fsl_bin_path not in current_path:
    print(f"Adding FSL bin directory to PATH: {fsl_bin_path}")
    # Prepend FSL path to existing PATH
    os.environ['PATH'] = f"{fsl_bin_path}{os.pathsep}{current_path}"
    print("Updated PATH:", os.environ['PATH']) # Optional: print long path
else:
    print(f"FSL bin directory already found in PATH.")

# Optional: Verify FSLOUTPUTTYPE (often needed)
if 'FSLOUTPUTTYPE' not in os.environ:
    print("Setting FSLOUTPUTTYPE=NIFTI_GZ")
    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

# --- Verification (Optional but Recommended) ---
print("\n--- Verifying FSL command access from Python ---")
try:
    # Use 'which' command (common on Linux) to find the command via the updated PATH
    which_bet = subprocess.run(['which', 'bet'], capture_output=True, text=True, check=True)
    print(f"Found 'bet' executable at: {which_bet.stdout.strip()}")

    # Or try running the command directly with --version
    # bet_version = subprocess.run(['bet', '--version'], capture_output=True, text=True)
    # if bet_version.returncode == 0:
    #     print("Successfully executed 'bet --version'.")
    # else:
    #     print(f"Warning: 'bet --version' failed with code {bet_version.returncode}. Stderr: {bet_version.stderr.strip()}")

except FileNotFoundError:
    print("ERROR: 'which' command not found (should be available on Linux).")
except subprocess.CalledProcessError:
    print("ERROR: 'bet' command not found in the updated PATH. Check fsl_dir path and FSL installation.")
except Exception as e:
    print(f"An unexpected error occurred during verification: {e}")

print("-" * 50)

# --- Now you can proceed with the rest of your notebook/script ---
# Example: Any subsequent calls like subprocess.run(['bet', ...]) should now work.
# import pandas as pd
# ... your other imports and code ...

# --- Intensity Normalization Script (using FSLMATHS Z-score within mask) ---

import pandas as pd
import os
import subprocess
from glob import glob
import shutil
import time
import numpy as np # For checking std dev

# --- 1. Configuration ---
print("\n--- Configuration for Intensity Normalization ---")
project_dir = '/home/ecren/pd_project'

# --- Input from previous step ---
# We need the metadata file that contains paths to:
# 1. Registered T1w image in MNI space
# 2. Corresponding non-linear warp field
# 3. Corresponding ORIGINAL (subject-space) brain mask
input_metadata_path = os.path.join(project_dir, 'registered_metadata.csv')

# --- Output for this step ---
output_normalized_dir = os.path.join(project_dir, 'nifti_normalized_mni') # NEW directory
output_warped_masks_dir = os.path.join(project_dir, 'temp_warped_masks') # Store warped masks (can be temp)
output_metadata_path = os.path.join(project_dir, 'normalized_metadata.csv') # NEW metadata file

# --- FSL Configuration & Reference ---
try:
    fsldir = os.environ['FSLDIR']
except KeyError:
    print("ERROR: $FSLDIR environment variable not set. Cannot find FSL templates.")
    raise SystemExit("Exiting.")
# We need the MNI template brain as a reference for warping the mask
mni_template_brain = os.path.join(fsldir, 'data/standard/MNI152_T1_1mm_brain.nii.gz')
# -----------------------------------------
print(f"INFO: Input metadata --> {input_metadata_path}")
print(f"INFO: Output Normalized NIfTI directory --> {output_normalized_dir}")
print(f"INFO: Temporary Warped Masks directory --> {output_warped_masks_dir}")
print(f"INFO: Output metadata --> {output_metadata_path}")
print(f"INFO: MNI Reference for Mask Warping --> {mni_template_brain}")

if not os.path.exists(mni_template_brain):
    print(f"ERROR: MNI Brain Template not found: {mni_template_brain}"); raise SystemExit

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(f"NOTE: This script requires 'registered_mni_path', 'nonlinear_warp_path',")
print(f"      and 'brain_mask_path' (original mask) columns in {input_metadata_path}")
print(f"NOTE: Output files will be placed in {output_normalized_dir}")
print(f"      Ensure this directory is empty or okay to overwrite/add to.")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- Create necessary output directories ---
os.makedirs(output_normalized_dir, exist_ok=True)
os.makedirs(output_warped_masks_dir, exist_ok=True) # For intermediate warped masks

# --- 2. Load the Metadata from the Registration Step ---
print(f"\n--- Loading Registration Metadata: {input_metadata_path} ---")
try:
    registered_df = pd.read_csv(input_metadata_path)
    print(f"Loaded metadata for {len(registered_df)} subjects.")
    # Define required columns
    required_cols = ['registered_mni_path', 'nonlinear_warp_path', 'brain_mask_path']

    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in registered_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from metadata: {', '.join(missing_cols)}. Please merge metadata if necessary.")

    # Check for null values in required columns
    null_check = registered_df[required_cols].isnull().any()
    if null_check.any():
        print("WARNING: Input metadata contains rows with missing values in required columns.")
        print(null_check[null_check].index.tolist()) # Show which columns have nulls
        # Filter out rows with nulls in essential paths
        initial_count = len(registered_df)
        registered_df = registered_df.dropna(subset=required_cols).copy()
        print(f"Processing {len(registered_df)} subjects after removing {initial_count - len(registered_df)} rows with missing paths.")

except FileNotFoundError:
    print(f"ERROR: Metadata file not found at {input_metadata_path}")
    raise SystemExit("Exiting. Did the previous registration step run correctly?")
except Exception as e:
    print(f"ERROR loading registration metadata: {e}")
    raise SystemExit("Exiting.")

if registered_df.empty:
    raise SystemExit("ERROR: Registration metadata is empty or all entries have missing required paths.")


# --- 3. Define Intensity Normalization Function ---
print("\n--- Defining Intensity Normalization Function (using FSL) ---")

def normalize_intensity(registered_img_path, original_mask_path, warp_field_path,
                          output_norm_dir, output_warp_mask_dir, mni_ref_path):
    """
    Normalizes intensity of a registered image using Z-scoring within a brain mask
    that has been warped to MNI space. Uses FSL commands.

    Args:
        registered_img_path (str): Path to T1w image registered to MNI space.
        original_mask_path (str): Path to the ORIGINAL subject-space brain mask.
        warp_field_path (str): Path to the nonlinear warp field transforming subj -> MNI.
        output_norm_dir (str): Directory to save the final normalized image.
        output_warp_mask_dir (str): Directory to save the intermediate warped mask.
        mni_ref_path (str): Path to the MNI template (used as reference for warping mask).

    Returns:
        str: Path to the normalized image, or None on failure.
    """
    # --- Validate Inputs ---
    if any(pd.isna(p) for p in [registered_img_path, original_mask_path, warp_field_path]):
        return None
    if not all(os.path.exists(p) for p in [registered_img_path, original_mask_path, warp_field_path]):
        print(f"  ERROR: One or more input files not found for {os.path.basename(registered_img_path)}")
        if not os.path.exists(registered_img_path): print(f"   Missing: {registered_img_path}")
        if not os.path.exists(original_mask_path): print(f"   Missing: {original_mask_path}")
        if not os.path.exists(warp_field_path): print(f"   Missing: {warp_field_path}")
        return None

    try:
        # --- Define Output Filenames ---
        reg_base_filename = os.path.basename(registered_img_path)
        # Extract the core subject/session identifier part
        name_part = reg_base_filename.split('_space-MNI')[0]

        output_normalized_filename = f"{name_part}_space-MNI152NLin_res-1mm_desc-norm_T1w.nii.gz"
        output_warped_mask_filename = f"{name_part}_space-MNI152NLin_res-1mm_desc-brain_mask.nii.gz" # MNI-space mask

        output_normalized_path = os.path.join(output_norm_dir, output_normalized_filename)
        output_warped_mask_path = os.path.join(output_warp_mask_dir, output_warped_mask_filename)

        # --- Check if Final Output Already Exists ---
        if os.path.exists(output_normalized_path):
            # print(f"  INFO: Normalized file already exists: {output_normalized_filename}")
            return output_normalized_path

        # === Step 1: Warp the Original Brain Mask to MNI Space ===
        if not os.path.exists(output_warped_mask_path):
            # print(f"  Warping mask: {os.path.basename(original_mask_path)} -> MNI space")
            # Use nearest neighbor interpolation for masks!
            applywarp_mask_command = [
                'applywarp',
                '--in=' + original_mask_path,
                '--ref=' + mni_ref_path,
                '--warp=' + warp_field_path,
                '--out=' + output_warped_mask_path,
                '--interp=nn' # Crucial for masks
            ]
            try:
                maskwarp_result = subprocess.run(applywarp_mask_command, capture_output=True, text=True, timeout=300, check=False)
                if maskwarp_result.returncode != 0 or not os.path.exists(output_warped_mask_path):
                    print(f"  ERROR: applywarp failed for MASK {original_mask_path}.")
                    print(f"  Command: {' '.join(applywarp_mask_command)}")
                    print(f"  Stderr: {maskwarp_result.stderr.strip()}")
                    if os.path.exists(output_warped_mask_path): os.remove(output_warped_mask_path) # Clean failed output
                    return None
            except FileNotFoundError: print(f"  ERROR: 'applywarp' command not found."); raise
            except subprocess.TimeoutExpired: print(f"  ERROR: Mask warping timed out."); return None
            except Exception as e: print(f"  ERROR during mask warping: {e}"); return None
        # else: print(f"  INFO: Using existing warped mask: {output_warped_mask_filename}")

        # === Step 2: Calculate Mean and Std Dev within the Warped Mask ===
        mean_val, std_val = None, None
        try:
            # print(f"  Calculating stats for {os.path.basename(registered_img_path)} using warped mask")
            fslstats_command = [
                'fslstats',
                registered_img_path,
                '-k', output_warped_mask_path, # Use the MNI-space mask
                '-M', # Mean of non-zero voxels
                '-S'  # Std Dev of non-zero voxels
            ]
            stats_result = subprocess.run(fslstats_command, capture_output=True, text=True, timeout=120, check=True) # check=True here
            # Output is typically "mean std\n"
            stats_output = stats_result.stdout.strip().split()
            if len(stats_output) == 2:
                mean_val = float(stats_output[0])
                std_val = float(stats_output[1])
                # print(f"    Mean={mean_val:.4f}, StdDev={std_val:.4f}")
                # Check for zero standard deviation
                if std_val is None or np.isclose(std_val, 0):
                     print(f"  ERROR: Standard deviation within mask is zero or near-zero for {registered_img_path}. Cannot normalize.")
                     return None
            else:
                print(f"  ERROR: Unexpected output from fslstats for {registered_img_path}: {stats_result.stdout.strip()}")
                return None
        except FileNotFoundError: print(f"  ERROR: 'fslstats' command not found."); raise
        except subprocess.TimeoutExpired: print(f"  ERROR: fslstats timed out."); return None
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: fslstats failed for {registered_img_path}.")
            print(f"  Command: {' '.join(e.cmd)}")
            print(f"  Stderr: {e.stderr.strip()}")
            return None
        except ValueError:
            print(f"  ERROR: Could not convert fslstats output to float: {stats_output}")
            return None
        except Exception as e: print(f"  ERROR during fslstats: {e}"); return None

        # === Step 3: Apply Z-score Normalization using fslmaths ===
        try:
            # print(f"  Applying Z-score normalization: (Image - {mean_val:.2f}) / {std_val:.2f}")
            fslmaths_command = [
                'fslmaths',
                registered_img_path,
                '-sub', str(mean_val),   # Subtract mean
                '-div', str(std_val),    # Divide by std dev
                '-mas', output_warped_mask_path, # Mask again to zero out non-brain voxels explicitly
                output_normalized_path
            ]
            maths_result = subprocess.run(fslmaths_command, capture_output=True, text=True, timeout=300, check=False)

            if maths_result.returncode != 0 or not os.path.exists(output_normalized_path):
                print(f"  ERROR: fslmaths normalization failed for {registered_img_path}.")
                print(f"  Command: {' '.join(fslmaths_command)}")
                print(f"  Stderr: {maths_result.stderr.strip()}")
                if os.path.exists(output_normalized_path): os.remove(output_normalized_path)
                return None

            # print(f"  Successfully created normalized file: {output_normalized_filename}")
            return output_normalized_path

        except FileNotFoundError: print(f"  ERROR: 'fslmaths' command not found."); raise
        except subprocess.TimeoutExpired: print(f"  ERROR: fslmaths timed out."); return None
        except Exception as e: print(f"  ERROR during fslmaths: {e}"); return None

    except Exception as e:
        print(f"  UNEXPECTED ERROR processing {registered_img_path}: {e}")
        return None


# --- 4. Run Intensity Normalization Processing ---
print(f"\n--- Running Intensity Normalization for {len(registered_df)} scans ---")

results = []
try:
    for index, row in registered_df.iterrows():
        reg_path = row['registered_mni_path']
        mask_path = row['brain_mask_path'] # Original mask path
        warp_path = row['nonlinear_warp_path']

        # Call the normalization function
        result_path = normalize_intensity(
            registered_img_path=reg_path,
            original_mask_path=mask_path,
            warp_field_path=warp_path,
            output_norm_dir=output_normalized_dir,
            output_warp_mask_dir=output_warped_masks_dir, # Where to put warped masks
            mni_ref_path=mni_template_brain
        )
        results.append(result_path)
        # Optional: Progress indicator
        # if (index + 1) % 50 == 0:
        #      print(f"  Processed {index + 1}/{len(registered_df)}...")

except FileNotFoundError:
     print("\nCRITICAL ERROR: FSL command (applywarp, fslstats, or fslmaths) not found. Aborting processing.")
     raise SystemExit("Exiting.")
except Exception as e:
     print(f"\nUNEXPECTED ERROR during processing loop: {e}")

# Add the results as a new column
registered_df['normalized_mni_path'] = results

print("\n--- Intensity Normalization Finished ---")
# Success means the final normalized path is not None
successful_processing = registered_df['normalized_mni_path'].notna()
num_successful = successful_processing.sum()
print(f"Successfully created/found normalized files for {num_successful} / {len(registered_df)} entries.")


# --- 5. Handle Failures & Save Final Metadata ---
failed_processing_df = registered_df[~successful_processing]

if not failed_processing_df.empty:
    print(f"\n--- Entries ({len(failed_processing_df)}) with Missing/Failed Normalization ---")
    # Display relevant columns for failed entries
    print(failed_processing_df[['Subject', 'Image Data ID', 'Group', 'registered_mni_path']]) # Show input path
    final_df = registered_df.dropna(subset=['normalized_mni_path']).copy()
    print(f"\nProceeding with {len(final_df)} successfully normalized subjects.")
else:
    final_df = registered_df.copy()
    print("\nAll selected scans normalized/found successfully.")

# --- 6. Save Final Metadata for this step ---
print(f"\n--- Saving Normalization Metadata to: {output_metadata_path} ---")
if not final_df.empty:
    try:
        # Select relevant columns to save
        cols_to_save = list(registered_df.columns) # Start with all from input df
        if 'normalized_mni_path' not in cols_to_save: cols_to_save.append('normalized_mni_path')

        # Filter to columns actually present in final_df
        cols_to_save_existing = [col for col in cols_to_save if col in final_df.columns]

        final_df[cols_to_save_existing].to_csv(output_metadata_path, index=False)
        print(f"Saved normalization metadata ({len(final_df)} subjects).")
    except Exception as e:
        print(f"ERROR saving normalization metadata: {e}")
else:
    print("WARNING: No subjects successfully processed. Normalization metadata file not saved.")



print("\n--- Intensity Normalization Script Complete ---")
print("\nNEXT STEP: Feature Extraction using the Harvard-Oxford Atlas.")
print("The normalized images are now in:", output_normalized_dir)