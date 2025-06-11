from exm.args.args import Args
from exm.puncta.basecalling import puncta_assign_gene, puncta_assign_nuclei

# Step 1: Load Configuration Settings
# ====================================

# Initialize the Args object for configuration.
args = Args()

# Provide the path to the configuration file.
args_file_path = '/orcd/data/edboyden/002/davy/ExSeq-Toolbox/ExSeq_toolbox_args.json'
# Load the configuration settings from the file.
args.load_params(args_file_path)

# Ensure the configuration file path is correct to avoid inconsistencies during processing.

# Step 2: Assign Genes to Detected Puncta for All FOVs
# ====================================================

operation_mode = 'original'  # Can be 'original' or 'improved'. Adjust as required.

for fov_for_gene_assignment in args.fovs:
    puncta_assign_gene(args=args, 
                       fov=fov_for_gene_assignment, 
                       option=operation_mode)

# Step 3: Assign Puncta to Nuclei for All FOVs
# ============================================

# Parameters for assigning puncta to nuclei.
distance_thresh = 100  # Threshold distance for assigning puncta to nuclei.
compare_to_surface = True  # Whether distance is computed to the nuclei surface.
nearest_nuclei_count = 3  # Number of nearest nuclei to consider.

for fov_for_nuclei_assignment in args.fovs:
    puncta_assign_nuclei(args=args, 
                         fov=fov_for_nuclei_assignment, 
                         distance_threshold=distance_thresh, 
                         compare_to_nuclei_surface=compare_to_surface, 
                         num_nearest_nuclei=nearest_nuclei_count, 
                         option=operation_mode)

# Note: Monitor logs or console output for any errors during processing.
