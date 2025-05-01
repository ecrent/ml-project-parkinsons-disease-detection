# --- DICOM Conversion Script  ---

import pandas as pd
import os
import subprocess
from glob import glob
import shutil
import nibabel as nib 


# --- 1. Configuration ---
print("\n--- Configuration for DICOM to NIfTI Conversion ---")
project_dir = '/home/ecren/projects/PPMI' # Adjust as needed
sampled_metadata_path = os.path.join(project_dir, '/workspaces/codespaces-jupyter/data/sampled_cohort_metadata_MG.csv')
# Directory containing the raw DICOM data (e.g., .../PPMI_DATA/subject_id/...)
image_base_dir = os.path.join('/workspaces/codespaces-jupyter/data/')

# --- Use DISTINCT output directory for CONVERTED NIfTI files ---
converted_nifti_dir = os.path.join(project_dir, 'nifti_converted') # Direct output dir
converted_metadata_path = os.path.join(project_dir, 'converted_metadata.csv') # Metadata for this step
# -----------------------------------------
print(f"INFO: Raw DICOM base --> {image_base_dir}")
print(f"INFO: NIfTI output --> {converted_nifti_dir}")
print(f"INFO: Output metadata --> {converted_metadata_path}")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(f"NOTE: This script will place NIfTI files in {converted_nifti_dir}")
print("      Ensure this directory is empty or okay to overwrite/add to.")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- Create necessary output directory ---
os.makedirs(converted_nifti_dir, exist_ok=True)

# --- 2. Load the Sampled Cohort Metadata ---
print(f"\n--- Loading Sampled Metadata: {sampled_metadata_path} ---")
try:
    subjects_to_process_df = pd.read_csv(sampled_metadata_path)
    print(f"Loaded metadata for {len(subjects_to_process_df)} subjects.")
    if 'Subject' not in subjects_to_process_df.columns or 'Image Data ID' not in subjects_to_process_df.columns:
        raise ValueError("Required columns 'Subject' and 'Image Data ID' missing.")
except FileNotFoundError:
    print(f"ERROR: Metadata file not found at {sampled_metadata_path}")
    raise SystemExit("Exiting.")
except Exception as e:
    print(f"ERROR loading sampled metadata: {e}")
    raise SystemExit("Exiting.")

if subjects_to_process_df.empty:
    raise SystemExit("ERROR: Sampled metadata is empty.")


# --- 3. Define DICOM Conversion Function ---
print("\n--- Defining DICOM Conversion Function ---")
def find_and_convert_dicom(subject_id, image_data_id, base_image_dir,
                           output_nifti_dir):
    """
    Finds the DICOM directory and converts it to NIfTI using dcm2niix.
    Returns the path to the created NIfTI file or None on failure.
    """
    try:
        # Ensure IDs are strings for path joining and filenames
        subject_id_str = str(int(subject_id))
        image_data_id_str = str(image_data_id)

        # --- Find the specific DICOM directory ---
        subject_base_path = os.path.join(base_image_dir, subject_id_str)
        if not os.path.isdir(subject_base_path):
            # print(f"  INFO: Subject directory not found: {subject_base_path}")
            return None

        # Search recursively within the subject directory for the Image Data ID directory
        search_pattern = os.path.join(subject_base_path, '**', image_data_id_str)
        potential_dirs = [
            d for d in glob(search_pattern, recursive=True)
            if os.path.isdir(d) and os.path.basename(d) == image_data_id_str
        ]

        if len(potential_dirs) == 0:
            # print(f"  INFO: No directory matching Image Data ID {image_data_id_str} found for Subject {subject_id_str}")
            return None
        elif len(potential_dirs) > 1:
            print(f"  WARNING: Multiple directories found for Subject {subject_id_str}, Image ID {image_data_id_str}. Using first: {potential_dirs[0]}")
            # Consider adding stricter error handling here if duplicates are unexpected
            dicom_dir = potential_dirs[0]
        else:
            dicom_dir = potential_dirs[0]
            # print(f"  Found DICOM directory: {dicom_dir}")

        # --- Define Output NIfTI filename ---
        # Simple, descriptive filename based on IDs
        output_filename_base = f"{subject_id_str}_{image_data_id_str}_T1w"
        # Target path (dcm2niix might add suffixes, so we'll find the actual output)
        target_output_path_prefix = os.path.join(output_nifti_dir, output_filename_base)
        final_output_path = target_output_path_prefix + ".nii.gz" # The ideal final name

        # --- Check if final output already exists ---
        # If the final .nii.gz exists, assume successful conversion previously
        if os.path.exists(final_output_path):
            # print(f"  INFO: Final NIfTI already exists: {final_output_path}")
            return final_output_path

        # --- Step 1: DICOM to NIfTI Conversion ---
        try:
            print(f"  Converting Subject {subject_id_str}, Image ID {image_data_id_str}...")
            # Define dcm2niix command
            # -o : Output directory
            # -f : Output filename format (%p=protocol, %s=series#, %d=description, %i=ID, %z=sequence)
            #      We'll use our desired base name directly.
            # -z y: Compress output (nii.gz)
            # -ba n: Do NOT anonymize BIDS sidecar (preserves potential info if needed later)
            # -b n: Do NOT generate BIDS sidecar (optional, can turn on if useful)
            # --progress y : Show progress bar
            # --verbose n : Reduce extra output
            command = ['dcm2niix',
                       '-o', output_nifti_dir,
                       '-f', output_filename_base,
                       '-z', 'y',
                       '-ba', 'n',
                       '-b', 'n',
                       '--progress', 'y',
                       '--verbose', 'n',
                       dicom_dir]

            # Run dcm2niix
            result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False) # check=False to handle errors manually

            # Check dcm2niix result
            if result.returncode != 0:
                # Check for common non-fatal warnings vs actual errors if possible
                # E.g., "Warning: Skipping Series VR (SI) is (US)" is often ignorable
                # if "error" in result.stderr.lower() or "fail" in result.stderr.lower(): # Basic check
                print(f"  ERROR: dcm2niix failed for {subject_id_str}, {image_data_id_str}.")
                print(f"  Stderr: {result.stderr.strip()}")
                return None
                # else:
                #     print(f"  WARNING: dcm2niix completed with non-zero code {result.returncode} but possibly only warnings. Continuing.")
                #     print(f"  Stderr: {result.stderr.strip()}")


            # --- Find the exact output file created by dcm2niix ---
            # dcm2niix might add sequence info or other details to the filename.
            # We specified '-f output_filename_base', so it *should* be predictable,
            # but globbing is safer.
            created_files = glob(f"{target_output_path_prefix}*.nii.gz")

            if not created_files:
                print(f"  ERROR: dcm2niix ran for {output_filename_base} but no output NIfTI file found matching pattern.")
                return None
            elif len(created_files) > 1:
                 print(f"  WARNING: Multiple NIfTI files found for {output_filename_base}. Using first: {created_files[0]}")
                 actual_converted_path = created_files[0]
            else:
                 actual_converted_path = created_files[0]

            # Optional: Rename to the exact desired filename if it differs
            if actual_converted_path != final_output_path:
                try:
                    print(f"  Renaming {os.path.basename(actual_converted_path)} to {os.path.basename(final_output_path)}")
                    os.rename(actual_converted_path, final_output_path)
                    return final_output_path
                except OSError as e:
                    print(f"  ERROR: Failed to rename {actual_converted_path} to {final_output_path}: {e}")
                    # Return the path that *was* created, even if renaming failed
                    return actual_converted_path
            else:
                 # print(f"  Successfully created NIfTI: {final_output_path}")
                 return final_output_path # Return the final path

        except FileNotFoundError:
             print(f"  ERROR: dcm2niix command not found. Is it installed and in your WSL PATH?")
             return None
        except subprocess.TimeoutExpired:
             print(f"  ERROR: dcm2niix timed out for {subject_id_str}, {image_data_id_str}.")
             return None
        except Exception as e:
             print(f"  ERROR during dcm2niix execution for {subject_id_str}: {e}")
             return None

    except Exception as e:
        # Catch unexpected errors during path manipulation or searching
        print(f"  UNEXPECTED ERROR processing Subject {subject_id}, Image ID {image_data_id}: {e}")
        return None


# --- 4. Run Conversion Processing ---
print(f"\n--- Running DICOM to NIfTI Conversion for {len(subjects_to_process_df)} scans ---")

results = []
for index, row in subjects_to_process_df.iterrows():
    result_path = find_and_convert_dicom(
        row['Subject'], row['Image Data ID'], image_base_dir,
        converted_nifti_dir
    )
    results.append(result_path)
    # Optional: Add a small delay or progress update if needed
    # if (index + 1) % 50 == 0:
    #     print(f"  Processed {index + 1}/{len(subjects_to_process_df)}...")

# Add the results as a new column
subjects_to_process_df['converted_nifti_path'] = results

print("\n--- Conversion Finished ---")
successful_processing = subjects_to_process_df['converted_nifti_path'].notna().sum()
print(f"Successfully processed/found NIfTI files for {successful_processing} / {len(subjects_to_process_df)} entries.")


# --- 5. Handle Failures & Save Final Metadata ---
failed_processing_df = subjects_to_process_df[subjects_to_process_df['converted_nifti_path'].isna()]
if not failed_processing_df.empty:
    print(f"\n--- Entries ({len(failed_processing_df)}) with Missing/Failed Conversion ---")
    # Display relevant columns for failed entries
    print(failed_processing_df[['Subject', 'Image Data ID', 'Group']]) # Add other relevant columns if needed
    final_df = subjects_to_process_df.dropna(subset=['converted_nifti_path']).copy()
    print(f"\nProceeding with {len(final_df)} successfully converted/found subjects.")
else:
    final_df = subjects_to_process_df.copy()
    print("\nAll selected scans processed/found successfully.")

# --- 6. Save Final Metadata ---
print(f"\n--- Saving Conversion Metadata to: {converted_metadata_path} ---")
if not final_df.empty:
    try:
        # Ensure the path column is included correctly
        if 'converted_nifti_path' not in final_df.columns:
             print("ERROR: 'converted_nifti_path' column missing before saving.")
        else:
             final_df.to_csv(converted_metadata_path, index=False)
             print(f"Saved conversion metadata ({len(final_df)} subjects).")
    except Exception as e:
        print(f"ERROR saving conversion metadata: {e}")
else:
    print("WARNING: No subjects successfully processed. Conversion metadata file not saved.")


# --- 7. Optional Cleanup ---
# You might want to remove any temporary files if dcm2niix created unexpected ones,
# although the current setup tries to manage this with specific output names.

print("\n--- DICOM Conversion Script Complete ---")
print(f"NEXT STEP: Visually inspect several NIfTI files in '{converted_nifti_dir}'")
print("           Check orientation and ensure they look correct before proceeding.")