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


import pandas as pd
import os
import subprocess
from glob import glob
import shutil

# --- 1. Configuration ---
print("\n--- Configuration for Orientation Standardization ---")
project_dir = '/workspaces/codespaces-jupyter'

# --- Input from previous step ---
input_nifti_dir = os.path.join(project_dir, 'nifti_converted')
input_metadata_path = os.path.join(project_dir, 'converted_metadata.csv')

# --- Output for this step ---
output_reoriented_dir = os.path.join(project_dir, 'nifti_reoriented') # NEW directory
output_metadata_path = os.path.join(project_dir, 'reoriented_metadata.csv') # NEW metadata file
# -----------------------------------------
print(f"INFO: Input NIfTI directory --> {input_nifti_dir}")
print(f"INFO: Input metadata --> {input_metadata_path}")
print(f"INFO: Output NIfTI directory --> {output_reoriented_dir}")
print(f"INFO: Output metadata --> {output_metadata_path}")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(f"NOTE: This script will place reoriented NIfTI files in {output_reoriented_dir}")
print("      Ensure this directory is empty or okay to overwrite/add to.")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- Create necessary output directory ---
os.makedirs(output_reoriented_dir, exist_ok=True)

# --- 2. Load the Metadata from the Conversion Step ---
print(f"\n--- Loading Conversion Metadata: {input_metadata_path} ---")
try:
    converted_df = pd.read_csv(input_metadata_path)
    print(f"Loaded metadata for {len(converted_df)} subjects.")
    if 'converted_nifti_path' not in converted_df.columns:
        raise ValueError("Required column 'converted_nifti_path' missing.")
    # Ensure the path column doesn't have missing values where processing should occur
    if converted_df['converted_nifti_path'].isnull().any():
        print("WARNING: Input metadata contains rows with missing 'converted_nifti_path'. These will be skipped.")
        # Optionally filter them out now or handle them in the loop
        # converted_df = converted_df.dropna(subset=['converted_nifti_path']).copy()
        # print(f"Processing {len(converted_df)} subjects with valid paths.")

except FileNotFoundError:
    print(f"ERROR: Metadata file not found at {input_metadata_path}")
    raise SystemExit("Exiting. Did the previous conversion step run correctly?")
except Exception as e:
    print(f"ERROR loading conversion metadata: {e}")
    raise SystemExit("Exiting.")

if converted_df.empty:
    raise SystemExit("ERROR: Conversion metadata is empty.")


# --- 3. Define Reorientation Function ---
print("\n--- Defining Orientation Standardization Function ---")
def reorient_nifti(input_nifti_path, output_dir):
    """
    Reorients a NIfTI file to standard orientation using fslreorient2std.
    Returns the path to the reoriented file or None on failure.
    """
    # --- Validate Input Path ---
    if pd.isna(input_nifti_path) or not isinstance(input_nifti_path, str):
        # print(f"  INFO: Skipping invalid input path: {input_nifti_path}")
        return None
    if not os.path.exists(input_nifti_path):
        print(f"  ERROR: Input NIfTI file not found: {input_nifti_path}")
        return None

    try:
        # --- Construct Output Path ---
        base_filename = os.path.basename(input_nifti_path)
        # Make a slightly modified name for the output file
        name, ext = os.path.splitext(base_filename)
        if name.endswith('.nii'): # Handle cases like .nii.gz
             name, _ = os.path.splitext(name)
        output_filename = f"{name}_reoriented.nii.gz" # Add suffix
        output_reoriented_path = os.path.join(output_dir, output_filename)

        # --- Check if output already exists ---
        if os.path.exists(output_reoriented_path):
            # print(f"  INFO: Reoriented file already exists: {output_reoriented_path}")
            return output_reoriented_path

        # --- Run fslreorient2std ---
        # print(f"  Reorienting: {os.path.basename(input_nifti_path)} -> {output_filename}")
        command = ['fslreorient2std', input_nifti_path, output_reoriented_path]

        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False) # check=False to handle manually

            if result.returncode != 0:
                print(f"  ERROR: fslreorient2std failed for {input_nifti_path}.")
                print(f"  Stderr: {result.stderr.strip()}")
                # Clean up potentially incomplete output file
                if os.path.exists(output_reoriented_path):
                    os.remove(output_reoriented_path)
                return None

            # --- Verify Output Creation ---
            if not os.path.exists(output_reoriented_path):
                 print(f"  ERROR: fslreorient2std ran for {input_nifti_path} but output file missing: {output_reoriented_path}")
                 return None

            # print(f"  Successfully created: {output_reoriented_path}")
            return output_reoriented_path

        except FileNotFoundError:
             print(f"  ERROR: fslreorient2std command not found. Is FSL installed and in your WSL PATH?")
             # Indicate failure for all subsequent files if the command is missing
             raise # Re-raise the exception to stop the script if the tool isn't found
        except subprocess.TimeoutExpired:
             print(f"  ERROR: fslreorient2std timed out for {input_nifti_path}.")
             return None
        except Exception as e:
             print(f"  ERROR during fslreorient2std execution for {input_nifti_path}: {e}")
             return None

    except Exception as e:
        # Catch unexpected errors during path manipulation etc.
        print(f"  UNEXPECTED ERROR processing {input_nifti_path}: {e}")
        return None


# --- 4. Run Reorientation Processing ---
print(f"\n--- Running Orientation Standardization for {len(converted_df)} scans ---")

results = []
try:
    for index, row in converted_df.iterrows():
        # Pass the path from the 'converted_nifti_path' column
        input_path = row['converted_nifti_path']
        result_path = reorient_nifti(input_path, output_reoriented_dir)
        results.append(result_path)
        # Optional: Progress indicator
        # if (index + 1) % 50 == 0:
        #      print(f"  Processed {index + 1}/{len(converted_df)}...")

except FileNotFoundError:
     # Catch the re-raised exception if fslreorient2std is missing
     print("\nCRITICAL ERROR: fslreorient2std not found. Aborting processing.")
     raise SystemExit("Exiting.")

# Add the results as a new column
converted_df['reoriented_nifti_path'] = results

print("\n--- Reorientation Finished ---")
successful_processing = converted_df['reoriented_nifti_path'].notna().sum()
print(f"Successfully created/found reoriented files for {successful_processing} / {len(converted_df)} entries.")


# --- 5. Handle Failures & Save Final Metadata ---
failed_processing_df = converted_df[converted_df['reoriented_nifti_path'].isna()]
if not failed_processing_df.empty:
    print(f"\n--- Entries ({len(failed_processing_df)}) with Missing/Failed Reorientation ---")
    # Display relevant columns for failed entries
    print(failed_processing_df[['Subject', 'Image Data ID', 'Group', 'converted_nifti_path']]) # Show input path too
    final_df = converted_df.dropna(subset=['reoriented_nifti_path']).copy()
    print(f"\nProceeding with {len(final_df)} successfully reoriented subjects.")
else:
    final_df = converted_df.copy()
    print("\nAll selected scans reoriented/found successfully.")

print(f"\n--- Saving Reorientation Metadata to: {output_metadata_path} ---")
if not final_df.empty:
    try:
        # final_df already contains the original columns loaded into converted_df
        # plus the 'reoriented_nifti_path' column (failed rows removed).
        # Simply save final_df directly.
        final_df.to_csv(output_metadata_path, index=False)
        print(f"Saved reorientation metadata ({len(final_df)} subjects).")
    except Exception as e:
        print(f"ERROR saving reorientation metadata: {e}")
else:
    print("WARNING: No subjects successfully processed. Reorientation metadata file not saved.")



# --- 7. Optional Cleanup ---
# No specific cleanup needed here unless intermediate files were created unexpectedly.

print("\n--- Orientation Standardization Script Complete ---")
print(f"NEXT STEP: Visually inspect several NIfTI files in '{output_reoriented_dir}'")
print("           Use a viewer like fsleyes. ALL images should now display")
print("           in the SAME orientation (e.g., axial, sagittal views look consistent).")
print("           Compare them to a standard template like MNI152.")