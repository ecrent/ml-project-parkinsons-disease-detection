import os
import matplotlib.pyplot as plt
from nilearn import plotting, image
import nibabel as nib # To potentially check image details

# --- 1. Configuration: SET THESE PATHS ---

project_dir = '/workspaces/codespaces-jupyter' # Or '/workspaces/codespaces-jupyter'

# --- Paths to files from PREVIOUS steps ---

# Get paths from your 'brain_extracted_metadata.csv' file
# You need the reoriented image (input to BET) and the mask (output from BET)
# for the SAME subject.



# Construct expected filenames based on previous scripts' naming conventions
reoriented_filename = f"/workspaces/codespaces-jupyter/nifti_reoriented/3808_I269578_T1w_reoriented.nii.gz"
mask_filename = f"/workspaces/codespaces-jupyter/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz" # Mask from BET

reoriented_mri_file = os.path.join(project_dir, 'nifti_reoriented', reoriented_filename)
brain_mask_file = os.path.join(project_dir, 'nifti_brain_extracted', mask_filename) # Mask is in brain_extracted dir

# --- Output file for the plot (optional) ---
output_plot_dir = os.path.join(project_dir, 'qc_plots')
os.makedirs(output_plot_dir, exist_ok=True)
# Create a filename based on the background image
bg_basename = os.path.basename(reoriented_mri_file).split('.')[0]
output_plot_file = os.path.join(output_plot_dir, f'overlay_brain_mask_on_{bg_basename}.png')

# -----------------------------------------

print(f"--- Configuration ---")
print(f"Background (Reoriented) MRI: {reoriented_mri_file}")
print(f"Overlay (Brain Mask) File: {brain_mask_file}")
print(f"Output plot directory: {output_plot_dir}")

# --- 2. Check if files exist ---
if not os.path.exists(reoriented_mri_file):
    print(f"ERROR: Reoriented MRI file not found at: {reoriented_mri_file}")
    print(f"       Check subject/image ID and 'nifti_reoriented' directory.")
    exit()
if not os.path.exists(brain_mask_file):
    print(f"ERROR: Brain mask file not found at: {brain_mask_file}")
    print(f"       Check subject/image ID and 'nifti_brain_extracted' directory. Did BET run successfully?")
    exit()

# --- 3. Create the Overlay Plot ---

# Load images (optional, plotting functions can take paths)
try:
    bg_img = image.load_img(reoriented_mri_file)
    mask_img = image.load_img(brain_mask_file)
    print(f"Background image shape: {bg_img.shape}, dtype: {bg_img.get_data_dtype()}")
    print(f"Mask image shape: {mask_img.shape}, dtype: {mask_img.get_data_dtype()}")

    # Ensure shapes match (they should if mask was derived from the bg_img)
    if bg_img.shape != mask_img.shape:
        print(f"WARNING: Background shape {bg_img.shape} and mask shape {mask_img.shape} differ!")
        # Consider resampling mask to background if necessary, but this indicates a potential problem
        # print("Attempting to resample mask...")
        # mask_img = image.resample_to_img(mask_img, bg_img, interpolation='nearest')

except Exception as e:
    print(f"Error loading NIfTI files: {e}")
    exit()

print("\nGenerating overlay plot...")

# --- Use plot_roi to show mask contours or filled overlay ---

# Option 1: Show mask as contours (often clearer for QC)
display = plotting.plot_anat(bg_img,
                              title=f"Brain Mask Contour on Reoriented Image\n{os.path.basename(reoriented_mri_file)}",
                              display_mode='ortho', # Ortho view is good for QC
                              cut_coords=None, # Let nilearn choose center
                              black_bg=True)
display.add_contours(mask_img,
                     levels=[0.5], # Threshold for contour drawing (0.5 for binary mask)
                     colors='r') # Color of the contour (e.g., red)

# Option 2: Show mask as filled overlay (Comment out Option 1 if using this)
# display = plotting.plot_roi(
#     roi_img=mask_img,
#     bg_img=bg_img,
#     title=f"Brain Mask Overlay on Reoriented Image\n{os.path.basename(reoriented_mri_file)}",
#     display_mode='ortho', # Ortho view is good for QC
#     cut_coords=None, # Let nilearn choose center
#     cmap='autumn', # Use a colormap that highlights the mask (e.g., red/yellow)
#     alpha=0.5, # Transparency
#     black_bg=True,
#     threshold=0.1 # Don't plot background (value 0)
# )

print("Plot generated.")


# --- 4. Save and Show Plot ---
try:
    print(f"Saving plot to: {output_plot_file}")
    plt.savefig(output_plot_file, dpi=200, bbox_inches='tight')
    print("Plot saved successfully.")
except Exception as e:
    print(f"Warning: Could not save plot: {e}")

print("Displaying plot...")
plotting.show() # This will attempt to display the plot

print("\nScript finished.")