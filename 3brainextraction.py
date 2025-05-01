# --- Brain Extraction Script (using FSL BET) ---

import os
import sys
import subprocess
import pandas as pd
from glob import glob
import shutil
import numpy as np # For NaN checks if needed

# --- Block 1: FSL Environment Setup (Crucial for Codespaces/Jupyter) ---
#region FSL Environment Setup
print("--- Setting FSL Environment Variables for Python ---")

# --- IMPORTANT: Verify this path matches your FSL installation ---
fsl_dir = "/workspaces/codespaces-jupyter/fsl" # MODIFY IF YOUR FSL PATH IS DIFFERENT
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
if fsl_bin_path not in current_path.split(os.pathsep): # Check components properly
    print(f"Adding FSL bin directory to PATH: {fsl_bin_path}")
    # Prepend FSL path to existing PATH
    os.environ['PATH'] = f"{fsl_bin_path}{os.pathsep}{current_path}"
    # print("Updated PATH:", os.environ['PATH']) # Optional: print long path
else:
    print(f"FSL bin directory already found in PATH.")

# Set FSLOUTPUTTYPE (often needed, ensures output is NIFTI_GZ)
print("Setting FSLOUTPUTTYPE=NIFTI_GZ")
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

# --- Verification (Optional but Recommended) ---
print("\n--- Verifying FSL command access from Python ---")
try:
    which_bet = subprocess.run(['which', 'bet'], capture_output=True, text=True, check=True, timeout=5)
    print(f"Found 'bet' executable at: {which_bet.stdout.strip()}")
except FileNotFoundError:
    print("ERROR: 'which' command not found (should be available on Linux).")
    sys.exit("Cannot verify FSL path.")
except subprocess.TimeoutExpired:
    print("ERROR: 'which bet' command timed out.")
    sys.exit("Cannot verify FSL path.")
except subprocess.CalledProcessError:
    print("ERROR: 'bet' command not found using 'which'. Check FSL installation and PATH setup.")
    sys.exit("FSL 'bet' command not accessible.")
except Exception as e:
    print(f"An unexpected error occurred during verification: {e}")
    sys.exit("Cannot verify FSL setup.")

print("-" * 50)
#endregion FSL Environment Setup


# --- Block 2: Script Configuration ---
#region Configuration
print("\n--- Configuration for Brain Extraction ---")
project_dir = '/workspaces/codespaces-jupyter' # MODIFY AS NEEDED

# --- Input from previous step ---
input_nifti_dir = os.path.join(project_dir, 'nifti_reoriented') # Not directly used, paths are from metadata
input_metadata_path = os.path.join(project_dir, 'reoriented_metadata.csv')

# --- Output for this step ---
output_brain_extracted_dir = os.path.join(project_dir, 'nifti_brain_extracted')
output_metadata_path = os.path.join(project_dir, 'brain_extracted_metadata.csv')

# --- BET Parameters ---
# *** ADJUST THIS THRESHOLD BASED ON VISUAL QC ***
# Common values for T1w: 0.3, 0.4, 0.5. Lower=bigger brain, Higher=smaller brain.
bet_f_threshold = 0.4
# ***********************************************

print(f"INFO: Project Directory --> {project_dir}")
print(f"INFO: Input Metadata --> {input_metadata_path}")
print(f"INFO: Output NIfTI Directory --> {output_brain_extracted_dir}")
print(f"INFO: Output Metadata --> {output_metadata_path}")
print(f"INFO: FSL BET Fractional Intensity Threshold (-f) = {bet_f_threshold}")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(f"NOTE: This script will place outputs in {output_brain_extracted_dir}")
print("      Ensure this directory is empty or okay to overwrite/add to.")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- Create necessary output directory ---
os.makedirs(output_brain_extracted_dir, exist_ok=True)
#endregion Configuration


# --- Block 3: Load Metadata ---
#region Load Metadata
print(f"\n--- Loading Reorientation Metadata: {input_metadata_path} ---")
try:
    reoriented_df = pd.read_csv(input_metadata_path)
    print(f"Loaded metadata for {len(reoriented_df)} subjects.")
    if 'reoriented_nifti_path' not in reoriented_df.columns:
        raise ValueError("Required column 'reoriented_nifti_path' missing from metadata.")
    if reoriented_df['reoriented_nifti_path'].isnull().any():
        print("WARNING: Input metadata contains rows with missing 'reoriented_nifti_path'.")
        initial_count = len(reoriented_df)
        reoriented_df = reoriented_df.dropna(subset=['reoriented_nifti_path']).copy()
        print(f"Processing {len(reoriented_df)} subjects after removing {initial_count - len(reoriented_df)} rows with missing paths.")

except FileNotFoundError:
    print(f"ERROR: Metadata file not found at {input_metadata_path}")
    sys.exit("Exiting. Did the previous reorientation step run correctly and save metadata?")
except ValueError as ve:
    print(f"ERROR: Problem with metadata columns: {ve}")
    sys.exit("Exiting.")
except Exception as e:
    print(f"ERROR loading reorientation metadata: {e}")
    sys.exit("Exiting.")

if reoriented_df.empty:
    print("ERROR: No subjects with valid paths found in the metadata file.")
    sys.exit("Exiting.")
#endregion Load Metadata


# --- Block 4: Define Brain Extraction Function ---
#region Brain Extraction Function
print("\n--- Defining Brain Extraction Function (using FSL BET) ---")
def extract_brain(input_nifti_path, output_dir, f_threshold):
    """
    Performs brain extraction using FSL BET.
    Generates both the brain-extracted image and the brain mask.
    Returns a tuple: (path_to_brain_image, path_to_brain_mask) or (None, None) on failure.
    """
    # --- Validate Input Path ---
    if pd.isna(input_nifti_path) or not isinstance(input_nifti_path, str):
        print(f"  WARNING: Invalid input path provided: {input_nifti_path}")
        return (None, None)
    if not os.path.exists(input_nifti_path):
        print(f"  ERROR: Input NIfTI file not found: {input_nifti_path}")
        return (None, None)

    try:
        # --- Construct Output Paths ---
        base_filename = os.path.basename(input_nifti_path)
        # Robustly remove extensions like .nii or .nii.gz
        if base_filename.endswith(".nii.gz"):
            name_part = base_filename[:-7] # Remove .nii.gz
        elif base_filename.endswith(".nii"):
            name_part = base_filename[:-4] # Remove .nii
        else:
            name_part = os.path.splitext(base_filename)[0] # Fallback

        output_brain_filename = f"{name_part}_brain.nii.gz"
        output_mask_filename = f"{name_part}_brain_mask.nii.gz"

        output_brain_path = os.path.join(output_dir, output_brain_filename)
        output_mask_path = os.path.join(output_dir, output_mask_filename)

        # --- Check if BOTH outputs already exist ---
        if os.path.exists(output_brain_path) and os.path.exists(output_mask_path):
            # print(f"  INFO: Outputs already exist for {base_filename}. Skipping BET.")
            return (output_brain_path, output_mask_path)

        # --- Build FSL BET Command ---
        command = [
            'bet',
            input_nifti_path,
            output_brain_path,
            '-f', str(f_threshold),
            '-R', # Robust mode
            '-m'  # Generate mask
        ]
        command_str = ' '.join(command) # For printing command on error

        # --- Run FSL BET ---
        print(f"  Running BET on: {base_filename}...")
        try:
            # Set a reasonable timeout (e.g., 5 minutes)
            result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)

            # --- Check for Errors after running ---
            if result.returncode != 0:
                print(f"  ERROR: FSL BET failed for {input_nifti_path} (Return Code: {result.returncode}).")
                print(f"  Command Run: {command_str}")
                # Print first few lines of stderr if available
                stderr_snippet = result.stderr.strip().splitlines()
                print(f"  Stderr (first 5 lines):\n    " + "\n    ".join(stderr_snippet[:5]))
                # Clean up potentially incomplete output files
                if os.path.exists(output_brain_path): os.remove(output_brain_path)
                if os.path.exists(output_mask_path): os.remove(output_mask_path)
                return (None, None)

            # --- Verify Output File Creation (Crucial) ---
            if not os.path.exists(output_brain_path) or not os.path.exists(output_mask_path):
                 print(f"  ERROR: FSL BET ran for {input_nifti_path} but one or both output files are missing!")
                 print(f"         Expected Brain: {output_brain_path} (Exists: {os.path.exists(output_brain_path)})")
                 print(f"         Expected Mask: {output_mask_path} (Exists: {os.path.exists(output_mask_path)})")
                 # Clean up partial results if any exist
                 if os.path.exists(output_brain_path): os.remove(output_brain_path)
                 if os.path.exists(output_mask_path): os.remove(output_mask_path)
                 return (None, None)

            print(f"  Successfully created: {output_brain_filename} and mask.")
            return (output_brain_path, output_mask_path)

        # --- Handle specific subprocess errors ---
        except subprocess.TimeoutExpired:
             print(f"  ERROR: FSL BET timed out for {input_nifti_path} (limit: 300s).")
             # Clean up potentially incomplete output files
             if os.path.exists(output_brain_path): os.remove(output_brain_path)
             if os.path.exists(output_mask_path): os.remove(output_mask_path)
             return (None, None)
        except Exception as sub_e: # Catch other potential errors during run
             print(f"  ERROR during FSL BET execution for {input_nifti_path}: {sub_e}")
             print(f"  Command attempted: {command_str}")
             return (None, None)

    except Exception as e:
        # Catch unexpected errors during path manipulation etc.
        print(f"  UNEXPECTED outer error processing {input_nifti_path}: {e}")
        return (None, None)
#endregion Brain Extraction Function


# --- Block 5: Run Processing Loop ---
#region Processing Loop
print(f"\n--- Running Brain Extraction for {len(reoriented_df)} scans ---")

results_list = [] # Store tuples of (brain_path, mask_path)
try:
    for index, row in reoriented_df.iterrows():
        print(f"Processing Subject {index+1}/{len(reoriented_df)}...")
        input_path = row['reoriented_nifti_path']
        # Pass the BET threshold parameter
        result_tuple = extract_brain(input_path, output_brain_extracted_dir, bet_f_threshold)
        results_list.append(result_tuple)

except FileNotFoundError:
     # This catches the re-raised exception if 'bet' command isn't found by subprocess
     print("\nCRITICAL ERROR: 'bet' command was not found by the subprocess.")
     print("Ensure FSL is correctly installed and the PATH is set up in Block 1.")
     sys.exit("Aborting processing.")
except Exception as loop_e:
     print(f"\nUNEXPECTED ERROR during processing loop at index {index}: {loop_e}")
     # Optionally decide whether to continue or stop
     # For now, we stop to investigate
     sys.exit("Aborting due to error in loop.")


# Add the results as new columns to the DataFrame
# Note: This assumes results_list has the same length as reoriented_df
if len(results_list) == len(reoriented_df):
    reoriented_df['brain_extracted_path'] = [res[0] for res in results_list]
    reoriented_df['brain_mask_path'] = [res[1] for res in results_list]
else:
    print("ERROR: Length mismatch between results and DataFrame rows. Cannot add result columns.")
    # Handle this error case - maybe fill with NaN or exit
    reoriented_df['brain_extracted_path'] = pd.NA
    reoriented_df['brain_mask_path'] = pd.NA

print("\n--- Brain Extraction Finished ---")

# Assess success (requires BOTH paths to be non-null)
if 'brain_extracted_path' in reoriented_df.columns and 'brain_mask_path' in reoriented_df.columns:
    successful_processing_mask = reoriented_df['brain_extracted_path'].notna() & reoriented_df['brain_mask_path'].notna()
    num_successful = successful_processing_mask.sum()
    print(f"Successfully created/found brain & mask files for {num_successful} / {len(reoriented_df)} entries.")
else:
    print("Warning: Result columns not added to DataFrame. Cannot assess success rate.")
    num_successful = 0
    successful_processing_mask = pd.Series([False] * len(reoriented_df)) # Assume failure

#endregion Processing Loop


# --- Block 6: Handle Failures & Save Final Metadata ---
#region Save Metadata
# Identify failures
failed_processing_df = reoriented_df[~successful_processing_mask]

if not failed_processing_df.empty:
    print(f"\n--- Entries ({len(failed_processing_df)}) with Missing/Failed Brain Extraction ---")
    # Display relevant columns for failed entries
    cols_to_show = ['Subject', 'Image Data ID', 'Group', 'reoriented_nifti_path'] # Assuming these exist
    cols_to_show = [col for col in cols_to_show if col in failed_processing_df.columns] # Check existence
    print(failed_processing_df[cols_to_show])

    # Keep only rows where both brain and mask paths are valid for the final dataframe
    final_df = reoriented_df[successful_processing_mask].copy()
    if not final_df.empty:
         print(f"\nProceeding with {len(final_df)} successfully brain-extracted subjects.")
    else:
         print("\nNo subjects were successfully processed.")
else:
    # If no failures, the final dataframe is the same as the input (with new columns)
    final_df = reoriented_df.copy()
    print("\nAll selected scans brain-extracted/found successfully.")

# --- Save the metadata for successfully processed subjects ---
print(f"\n--- Saving Brain Extraction Metadata to: {output_metadata_path} ---")
if not final_df.empty:
    try:
        # Ensure necessary columns exist before saving
        final_columns = [col for col in final_df.columns if col not in ['Unnamed: 0', 'index']] # Clean common issues
        final_df[final_columns].to_csv(output_metadata_path, index=False)
        print(f"Saved brain extraction metadata ({len(final_df)} subjects).")
    except Exception as e:
        print(f"ERROR saving brain extraction metadata: {e}")
else:
    print("WARNING: No subjects successfully processed. Brain extraction metadata file not saved.")
#endregion Save Metadata


# --- Block 7: Final Instructions ---
#region Final Instructions
print("\n--- Brain Extraction Script Complete ---")
print("\n!!!!!!!!!!!!!!!!!!!! IMPORTANT: QUALITY CONTROL !!!!!!!!!!!!!!!!!!!!")
print(f"Visually inspect MULTIPLE outputs in '{output_brain_extracted_dir}'.")
print("Use a viewer like fsleyes (you might need to download outputs locally):")
print("  1. Load an ORIGINAL reoriented image (from 'nifti_reoriented').")
print("  2. Overlay the corresponding BRAIN MASK (_brain_mask.nii.gz) from")
print(f"     '{output_brain_extracted_dir}' on top (e.g., as a red contour).")
print("Check if the mask accurately covers the brain without removing excessive")
print("brain tissue or including large chunks of skull/eyes/neck.")
print(f"If results are poor (mask too small/large), adjust 'bet_f_threshold' (currently {bet_f_threshold})")
print("near the top of this script (Block 2) and RE-RUN this script.")
print("(You may want to delete the contents of")
print(f"'{output_brain_extracted_dir}' before re-running).")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("\nNEXT STEP: Registration to Standard Space (MNI).")
#endregion Final Instructions