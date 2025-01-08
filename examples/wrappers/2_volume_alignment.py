from exm.args.args import Args
from exm.align.align import volumetric_alignment

# 1: Load the Configuration
# =================================

# Initialize the configuration object.
args = Args()

# Provide the path to the configuration file. Typically, this file is generated 
# during the pipeline configuration step and has a '.json' extension.
args_file_path = '/path/to/your/parameters/file.json'

# Load the configuration settings from the specified file.
args.load_params(args_file_path)

# Note: Ensure the provided path points to the correct configuration file 
# to avoid any inconsistencies during subsequent processing steps.

# 2: Set Parameters for Volumetric Alignment
# ==========================================

# Specify general parameters
parallelization = 4  # Number of parallel processes
background_subtraction = ''  # 'top_hat' or 'rolling_ball', or '' for no background subtraction

# Define alignment steps for RANSAC and Affine
# Downsample steps: Perform RANSAC on a downsampled volume with factors (2, 4, 4)
downsample_factors = (2, 4, 4)
downsample_steps = [
    ('ransac', {'blob_sizes': [5, 150], 'safeguard_exceptions': False})
]

# Full-size steps: Perform affine alignment
full_size_steps = [
    ('affine', {
        'metric': 'MMI',  
        'optimizer': 'LBFGSB',  
        'alignment_spacing': 1,  
        'shrink_factors': (4, 2, 1), 
        'smooth_sigmas': (0.0, 0.0, 0.0),
        'optimizer_args': {
            'gradientConvergenceTolerance': 1e-6,
            'numberOfIterations': 800,
            'maximumNumberOfCorrections': 8,
        }
    })
]

# Define additional parameters for normalization and downsample behavior
kwargs = {
    'downsample_factors': downsample_factors,
    'downsample_steps': downsample_steps,
    'full_size_steps': full_size_steps,
    'run_downsample_steps': True,  # Execute downsample alignment steps
    'low': 1.0,  # Low percentile for intensity normalization
    'high': 99.0  # High percentile for intensity normalization
}

# If you have specific Code-FOV pairs, specify them here. Otherwise, use all from config.
specific_code_fov_pairs = [(code_val, fov_val) for code_val in args.codes for fov_val in args.fovs]

# 3: Execute Volumetric Alignment
# ===============================
volumetric_alignment(
    args=args,
    code_fov_pairs=specific_code_fov_pairs,
    parallel_processes=parallelization,
    bg_sub=background_subtraction,
    accelerated=False,  # Set to True if GPU acceleration is available
    **kwargs  # Pass alignment parameters
)

# Note: Always monitor the logs or console output to ensure the alignment process 
# is proceeding without errors.
