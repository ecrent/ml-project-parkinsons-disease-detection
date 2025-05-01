# --- Registration to MNI Standard Space Script (using FSL FLIRT/FNIRT) ---

import pandas as pd
import os
import subprocess
from glob import glob
import shutil
import time

# --- 1. Configuration ---
print("\n--- Configuration for Registration to MNI Space ---")
project_dir = '/workspaces/codespaces-jupyter'

# --- Input from previous steps ---
input_nifti_dir = os.path.join(project_dir, 'nifti_brain_extracted') # Not directly used, paths are from metadata
input_metadata_path = os.path.join(project_dir, 'brain_extracted_metadata.csv')
# We also need the reoriented (full head) path for FNIRT warp calculation
# Assuming 'reoriented_nifti_path' column exists in brain_extracted_metadata.csv
# If not, you might need to merge metadata files first.

# --- Output for this step ---
output_registered_dir = os.path.join(project_dir, 'nifti_registered_mni') # NEW directory for final outputs
output_transforms_dir = os.path.join(project_dir, 'transforms_to_mni') # Store .mat and warp files
output_metadata_path = os.path.join(project_dir, 'registered_metadata.csv') # NEW metadata file

# --- FSL Configuration ---
try:
    fsldir = os.environ['FSLDIR']
except KeyError:
    print("ERROR: $FSLDIR environment variable not set. Cannot find FSL templates.")
    raise SystemExit("Exiting.")

# Standard MNI Templates (adjust path/resolution if needed)
mni_template_head = os.path.join(fsldir, 'data/standard/MNI152_T1_1mm.nii.gz')
mni_template_brain = os.path.join(fsldir, 'data/standard/MNI152_T1_1mm_brain.nii.gz')
# FNIRT Configuration file (standard one for T1w to MNI)
fnirt_config_file = os.path.join(fsldir, 'etc/flirtsch/T1_2_MNI152_1mm.cnf') # Check if this path is correct

# --- Registration Parameters ---
registration_mode = 'nonlinear' # Options: 'nonlinear' (FNIRT - recommended) or 'linear' (FLIRT only)
flirt_dof = 12 # Degrees of freedom for linear registration
flirt_cost_func = 'corratio' # Cost function for FLIRT (corratio often good for T1w)
# -----------------------------------------
print(f"INFO: Input metadata --> {input_metadata_path}")
print(f"INFO: Output registered NIfTI directory --> {output_registered_dir}")
print(f"INFO: Output transforms directory --> {output_transforms_dir}")
print(f"INFO: Output metadata --> {output_metadata_path}")
print(f"INFO: Registration Mode --> {registration_mode}")
print(f"INFO: MNI Template (Brain) --> {mni_template_brain}")
print(f"INFO: MNI Template (Head) --> {mni_template_head}")
if registration_mode == 'nonlinear':
     print(f"INFO: FNIRT Config --> {fnirt_config_file}")

# --- Validate FSL Files ---
if not os.path.exists(mni_template_head): print(f"ERROR: MNI Head Template not found: {mni_template_head}"); raise SystemExit
if not os.path.exists(mni_template_brain): print(f"ERROR: MNI Brain Template not found: {mni_template_brain}"); raise SystemExit
if registration_mode == 'nonlinear' and not os.path.exists(fnirt_config_file):
    print(f"ERROR: FNIRT Config file not found: {fnirt_config_file}"); raise SystemExit

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(f"NOTE: This script will place registered NIfTIs in {output_registered_dir}")
print(f"      and transformation files (.mat, warps) in {output_transforms_dir}")
print("      Ensure these directories are empty or okay to overwrite/add to.")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# --- Create necessary output directories ---
os.makedirs(output_registered_dir, exist_ok=True)
os.makedirs(output_transforms_dir, exist_ok=True)

# --- 2. Load the Metadata from the Brain Extraction Step ---
print(f"\n--- Loading Brain Extraction Metadata: {input_metadata_path} ---")
try:
    extracted_df = pd.read_csv(input_metadata_path)
    print(f"Loaded metadata for {len(extracted_df)} subjects.")
    required_cols = ['brain_extracted_path']
    # Need the original reoriented full head image for non-linear registration warp calculation
    if registration_mode == 'nonlinear':
        required_cols.append('reoriented_nifti_path')

    for col in required_cols:
        if col not in extracted_df.columns:
            raise ValueError(f"Required column '{col}' missing from metadata.")
        if extracted_df[col].isnull().any():
            print(f"WARNING: Input metadata contains rows with missing '{col}'. These will be skipped.")
            # extracted_df = extracted_df.dropna(subset=required_cols).copy()
            # print(f"Processing {len(extracted_df)} subjects with valid paths.")

except FileNotFoundError:
    print(f"ERROR: Metadata file not found at {input_metadata_path}")
    raise SystemExit("Exiting. Did the previous brain extraction step run correctly?")
except Exception as e:
    print(f"ERROR loading brain extraction metadata: {e}")
    raise SystemExit("Exiting.")

if extracted_df.empty:
    raise SystemExit("ERROR: Brain extraction metadata is empty or all entries have missing paths.")


# --- 3. Define Registration Function ---
print(f"\n--- Defining Registration Function (Mode: {registration_mode}) ---")

def register_to_mni(subject_brain_path, subject_reoriented_head_path, # head path only needed for nonlinear
                    output_reg_dir, output_xfm_dir,
                    mni_brain_ref, mni_head_ref, fnirt_config,
                    mode='nonlinear', flirt_options={'dof': 12, 'cost': 'corratio'}):
    """
    Registers a subject's T1w image to MNI space using FSL FLIRT and optionally FNIRT.

    Args:
        subject_brain_path (str): Path to the subject's brain-extracted T1w.
        subject_reoriented_head_path (str): Path to subject's reoriented full-head T1w (used for FNIRT).
        output_reg_dir (str): Directory to save the final registered image.
        output_xfm_dir (str): Directory to save transformation files (.mat, warp).
        mni_brain_ref (str): Path to the MNI brain template.
        mni_head_ref (str): Path to the MNI full-head template.
        fnirt_config (str): Path to the FNIRT configuration file.
        mode (str): 'nonlinear' or 'linear'.
        flirt_options (dict): Dictionary with 'dof' and 'cost' for FLIRT.

    Returns:
        tuple: (path_registered_image, path_linear_mat, path_warp_field)
               path_warp_field will be None if mode is 'linear'.
               Returns (None, None, None) on failure.
    """
    start_time = time.time()
    # --- Validate Inputs ---
    if pd.isna(subject_brain_path): return (None, None, None) # Basic check
    if not os.path.exists(subject_brain_path):
        print(f"  ERROR: Input brain file not found: {subject_brain_path}")
        return (None, None, None)
    if mode == 'nonlinear':
        if pd.isna(subject_reoriented_head_path): return (None, None, None)
        if not os.path.exists(subject_reoriented_head_path):
            print(f"  ERROR: Input reoriented head file not found (needed for FNIRT): {subject_reoriented_head_path}")
            return (None, None, None)

    try:
        # --- Define Output Filenames ---
        base_filename = os.path.basename(subject_brain_path)
        name_part = base_filename.split('_brain.')[0] # Get base name before _brain

        output_registered_filename = f"{name_part}_space-MNI152NLin_res-1mm_desc-preproc_T1w.nii.gz" # BIDS-like naming convention
        output_linear_mat_filename = f"{name_part}_from-T1w_to-MNI152NLin_mode-image_xfm.mat"
        output_warp_filename = f"{name_part}_from-T1w_to-MNI152NLin_mode-image_warp.nii.gz"

        output_registered_path = os.path.join(output_reg_dir, output_registered_filename)
        output_linear_mat_path = os.path.join(output_xfm_dir, output_linear_mat_filename)
        output_warp_path = os.path.join(output_xfm_dir, output_warp_filename)

        # --- Check if Final Output Already Exists ---
        if os.path.exists(output_registered_path):
            # print(f"  INFO: Final registered file already exists: {output_registered_filename}")
            # Return paths assuming transforms also exist (may need to check them too if needed)
            warp_path_to_return = output_warp_path if mode == 'nonlinear' and os.path.exists(output_warp_path) else None
            mat_path_to_return = output_linear_mat_path if os.path.exists(output_linear_mat_path) else None
            return (output_registered_path, mat_path_to_return, warp_path_to_return)

        # === Step 1: Linear Registration (FLIRT) ===
        # Always needed, either as final step or input to FNIRT
        # Only run if the .mat file doesn't exist
        if not os.path.exists(output_linear_mat_path):
            print(f"  Running FLIRT: {os.path.basename(subject_brain_path)} -> MNI")
            # Temporary output for FLIRT's registered image (we'll create the final one with applywarp if needed)
            temp_flirt_out = os.path.join(output_xfm_dir, f"{name_part}_tempflirt.nii.gz")
            flirt_command = [
                'flirt',
                '-in', subject_brain_path,
                '-ref', mni_brain_ref,
                '-out', temp_flirt_out, # Output linearly registered brain
                '-omat', output_linear_mat_path, # Output matrix
                '-dof', str(flirt_options['dof']),
                '-cost', flirt_options['cost']
            ]
            try:
                flirt_result = subprocess.run(flirt_command, capture_output=True, text=True, timeout=600, check=False)
                if flirt_result.returncode != 0 or not os.path.exists(output_linear_mat_path):
                    print(f"  ERROR: FLIRT failed for {subject_brain_path}.")
                    print(f"  Command: {' '.join(flirt_command)}")
                    print(f"  Stderr: {flirt_result.stderr.strip()}")
                    if os.path.exists(temp_flirt_out): os.remove(temp_flirt_out) # Clean temp file
                    if os.path.exists(output_linear_mat_path): os.remove(output_linear_mat_path)
                    return (None, None, None)
                # print(f"  FLIRT completed. Matrix: {output_linear_mat_filename}")
                # Clean up the temporary flirt output image if we are doing non-linear next
                if mode == 'nonlinear' and os.path.exists(temp_flirt_out):
                     os.remove(temp_flirt_out)
                elif mode == 'linear' and os.path.exists(temp_flirt_out):
                     # If only linear, rename temp file to final output
                     if os.path.exists(output_registered_path): os.remove(output_registered_path)
                     os.rename(temp_flirt_out, output_registered_path)

            except FileNotFoundError: print(f"  ERROR: 'flirt' command not found."); raise
            except subprocess.TimeoutExpired: print(f"  ERROR: FLIRT timed out."); return (None, None, None)
            except Exception as e: print(f"  ERROR during FLIRT: {e}"); return (None, None, None)
        # else: print(f"  INFO: Using existing FLIRT matrix: {output_linear_mat_filename}")


        # === Step 2: Non-linear Registration (FNIRT - Warp Calculation) ===
        if mode == 'nonlinear':
            # Calculate warp field if it doesn't exist
            if not os.path.exists(output_warp_path):
                print(f"  Running FNIRT (warp calc): {os.path.basename(subject_reoriented_head_path)} -> MNI")
                fnirt_command = [
                    'fnirt',
                    '--in=' + subject_reoriented_head_path, # Use full head image for warp calculation
                    '--ref=' + mni_head_ref,              # Use full head reference
                    '--aff=' + output_linear_mat_path,     # Use FLIRT matrix as starting point
                    '--cout=' + output_warp_path,         # Output warp field coefficient file
                    '--config=' + fnirt_config_file        # Configuration file
                    # Add other FNIRT options like --subsamp, --miter if needed/tuned
                ]
                try:
                    fnirt_result = subprocess.run(fnirt_command, capture_output=True, text=True, timeout=3600*2, check=False) # Long timeout
                    if fnirt_result.returncode != 0 or not os.path.exists(output_warp_path):
                        print(f"  ERROR: FNIRT (warp calculation) failed for {subject_reoriented_head_path}.")
                        print(f"  Command: {' '.join(fnirt_command)}")
                        print(f"  Stderr: {fnirt_result.stderr.strip()}")
                        if os.path.exists(output_warp_path): os.remove(output_warp_path)
                        return (None, output_linear_mat_path, None) # Return None for warp/final, but keep matrix path
                    # print(f"  FNIRT warp calculation complete.")
                except FileNotFoundError: print(f"  ERROR: 'fnirt' command not found."); raise
                except subprocess.TimeoutExpired: print(f"  ERROR: FNIRT timed out."); return (None, output_linear_mat_path, None)
                except Exception as e: print(f"  ERROR during FNIRT: {e}"); return (None, output_linear_mat_path, None)
            # else: print(f"  INFO: Using existing FNIRT warp: {output_warp_filename}")

            # === Step 3: Apply Warp (Non-linear only) ===
            # Apply the calculated warp field to the BRAIN image
            if os.path.exists(output_warp_path) and not os.path.exists(output_registered_path):
                 print(f"  Running applywarp: {os.path.basename(subject_brain_path)} -> MNI")
                 applywarp_command = [
                     'applywarp',
                     '--in=' + subject_brain_path,    # Input is the brain-extracted subject image
                     '--ref=' + mni_brain_ref,       # Reference is MNI brain template
                     '--warp=' + output_warp_path,    # Warp field from FNIRT
                     '--out=' + output_registered_path, # Final registered output image
                     '--interp=spline'                # Interpolation method (spline often good)
                 ]
                 try:
                      applywarp_result = subprocess.run(applywarp_command, capture_output=True, text=True, timeout=600, check=False)
                      if applywarp_result.returncode != 0 or not os.path.exists(output_registered_path):
                           print(f"  ERROR: applywarp failed for {subject_brain_path}.")
                           print(f"  Command: {' '.join(applywarp_command)}")
                           print(f"  Stderr: {applywarp_result.stderr.strip()}")
                           if os.path.exists(output_registered_path): os.remove(output_registered_path)
                           # Return None for final registered, but keep mat and warp paths if they exist
                           return (None, output_linear_mat_path, output_warp_path)
                      # print(f"  applywarp completed.")
                 except FileNotFoundError: print(f"  ERROR: 'applywarp' command not found."); raise
                 except subprocess.TimeoutExpired: print(f"  ERROR: applywarp timed out."); return (None, output_linear_mat_path, output_warp_path)
                 except Exception as e: print(f"  ERROR during applywarp: {e}"); return (None, output_linear_mat_path, output_warp_path)

        # --- Return Paths ---
        # If we got here, registration should be complete or used existing file
        final_warp_path = output_warp_path if mode == 'nonlinear' and os.path.exists(output_warp_path) else None
        final_reg_path = output_registered_path if os.path.exists(output_registered_path) else None
        final_mat_path = output_linear_mat_path if os.path.exists(output_linear_mat_path) else None

        if final_reg_path: # Check if the most critical output exists
             end_time = time.time()
             print(f"  Registration successful for {name_part} ({(end_time - start_time):.1f} s)")
             return (final_reg_path, final_mat_path, final_warp_path)
        else:
             print(f"  ERROR: Final registered file missing after processing {name_part}")
             return (None, final_mat_path, final_warp_path) # Return None for reg path

    except Exception as e:
        print(f"  UNEXPECTED ERROR processing {subject_brain_path}: {e}")
        return (None, None, None)


# --- 4. Run Registration Processing ---
print(f"\n--- Running Registration (Mode: {registration_mode}) for {len(extracted_df)} scans ---")

results = [] # Store tuples of (reg_path, mat_path, warp_path)
try:
    for index, row in extracted_df.iterrows():
        brain_path = row['brain_extracted_path']
        head_path = row.get('reoriented_nifti_path', None) # Use .get for safety if column might be missing

        # Call the registration function
        result_tuple = register_to_mni(
            subject_brain_path=brain_path,
            subject_reoriented_head_path=head_path,
            output_reg_dir=output_registered_dir,
            output_xfm_dir=output_transforms_dir,
            mni_brain_ref=mni_template_brain,
            mni_head_ref=mni_template_head,
            fnirt_config=fnirt_config_file,
            mode=registration_mode,
            flirt_options={'dof': flirt_dof, 'cost': flirt_cost_func}
        )
        results.append(result_tuple)
        # Optional: Progress indicator
        # if (index + 1) % 5 == 0: # Registration is slow, update more often
        #      print(f"  Processed {index + 1}/{len(extracted_df)}...")

except FileNotFoundError:
     print("\nCRITICAL ERROR: FSL command (flirt, fnirt, or applywarp) not found. Aborting processing.")
     raise SystemExit("Exiting.")
except Exception as e:
     print(f"\nUNEXPECTED ERROR during processing loop: {e}")
     # Decide whether to stop or continue

# Add the results as new columns
extracted_df['registered_mni_path'] = [res[0] for res in results]
extracted_df['linear_reg_mat_path'] = [res[1] for res in results]
extracted_df['nonlinear_warp_path'] = [res[2] for res in results] # Will be None if mode='linear'

print("\n--- Registration Finished ---")
# Success means the final registered path is not None
successful_processing = extracted_df['registered_mni_path'].notna()
num_successful = successful_processing.sum()
print(f"Successfully created/found registered files for {num_successful} / {len(extracted_df)} entries.")

# --- 5. Handle Failures & Save Final Metadata ---
failed_processing_df = extracted_df[~successful_processing]

if not failed_processing_df.empty:
    print(f"\n--- Entries ({len(failed_processing_df)}) with Missing/Failed Registration ---")
    print(failed_processing_df[['Subject', 'Image Data ID', 'Group', 'brain_extracted_path']]) # Show inputs
    final_df = extracted_df.dropna(subset=['registered_mni_path']).copy()
    print(f"\nProceeding with {len(final_df)} successfully registered subjects.")
else:
    final_df = extracted_df.copy()
    print("\nAll selected scans registered/found successfully.")

# --- 6. Save Final Metadata for this step ---
print(f"\n--- Saving Registration Metadata to: {output_metadata_path} ---")
if not final_df.empty:
    try:
        # Select relevant columns
        cols_to_save = list(extracted_df.columns) # Start with all from input df
        new_cols = ['registered_mni_path', 'linear_reg_mat_path', 'nonlinear_warp_path']
        for nc in new_cols:
             if nc not in cols_to_save: cols_to_save.append(nc)

        # Filter to columns actually present in final_df
        cols_to_save_existing = [col for col in cols_to_save if col in final_df.columns]

        final_df[cols_to_save_existing].to_csv(output_metadata_path, index=False)
        print(f"Saved registration metadata ({len(final_df)} subjects).")
    except Exception as e:
        print(f"ERROR saving registration metadata: {e}")
else:
    print("WARNING: No subjects successfully processed. Registration metadata file not saved.")

# --- 7. Optional Cleanup ---
# Consider removing temporary FLIRT outputs if desired, though the script handles this for nonlinear mode.

print("\n--- Registration Script Complete ---")
print("\n!!!!!!!!!!!!!!!!!!!! IMPORTANT: QUALITY CONTROL !!!!!!!!!!!!!!!!!!!!")
print(f"Visually inspect MULTIPLE outputs in '{output_registered_dir}'.")
print("Use a viewer like fsleyes and:")
print(f"  1. Load the MNI TEMPLATE brain ({os.path.basename(mni_template_brain)}) as the bottom layer.")
print("  2. Overlay a SUBJECT'S registered brain (..._space-MNI152NLin_..._T1w.nii.gz) on top.")
print("Check if the subject's brain contours and internal structures (ventricles,")
print("basal ganglia, cortical folds) align well with the template.")
print("Poor alignment might require tuning FLIRT/FNIRT parameters or indicate")
print("issues in previous steps (orientation, brain extraction).")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("\nNEXT STEP: Feature Extraction using the Harvard-Oxford Atlas.")