from exm.args.args import Args
from exm.align.align_eval import measure_round_alignment_NCC,plot_alignment_evaluation,calculate_alignment_evaluation_ci

# Step 1: Load Configuration Settings
# ====================================

# Create a new Config object instance.
args = Args()

# Provide the path to the configuration file.
args_file_path = '/path/to/your/parameter/file.json'

# Load the configuration settings from the specified file.
args.load_config(args_file_path)

# Step 2: Additional Configuration for Alignment Evaluation
# ================================================

# `nonzero_thresh`: This parameter specifies the threshold for the number of non-zero 
# pixels in a given volume. If the number of non-zero pixels in the aligned volume 
# is below this threshold, then the alignment evaluation for that volume is skipped.
args.nonzero_thresh = .2 * 2048 * 2048 * 80

# `N`: Number of random sub-volumes to sample for the NCC calculation.
args.N = 1000

# `subvol_dim`: Dimension (in pixels) of the cubic sub-volumes to sample for NCC calculation.
args.subvol_dim = 100

# `xystep`: Pixel size in the XY plane. Adjust based on microscope and imaging settings.
args.xystep = 0.1625/40 # check value

# `zstep`: Pixel size in the Z dimension. Adjust based on microscope and imaging settings.
args.zstep = 0.4/40 # check value

# `pct_thresh`: Percentile threshold value used for alignment evaluation. 
args.pct_thresh = 99


# Step 3: Alignment Measurement
# ===========================================================

# Define the list of Codes, Fovs number for which alignment will be evaluated.
codes_to_analyze = args.codes
fovs_to_analyze = args.fovs  # Adjust this based on your dataset for example [1,3].

# Extract the coordinates.
for fov in fovs_to_analyze:
    for code in codes_to_analyze:
        measure_round_alignment_NCC(args=args,code=code, fov=fov)


# Step 4: Alignment Evaluation and Confidence Interval Calculation
# ==========================================================

# Define CI and percentile parameters.
ci_percentage = 95
percentile_filter_value = 95

for fov in fovs_to_analyze:
    # Plot alignment evaluation
    plot_alignment_evaluation(args, fov, percentile=percentile_filter_value, save_fig=True)
    
    # Calculate alignment evaluation confidence interval (CI)
    calculate_alignment_evaluation_ci(args, fov, ci=ci_percentage, percentile_filter=percentile_filter_value)