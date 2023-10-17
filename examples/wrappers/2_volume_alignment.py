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

# 2: Execute Volumetric Alignment
# ====================================

# Specify additional parameters for alignment
parallelization = 4  # Number of parallel processes
alignment_method = 'bigstream'  # or None for SimpleITK
background_subtraction = ''  # 'top_hat' or 'rolling_ball' , or '' for no background subtraction  

# If you have specific round and ROI pairs you want to align, specify them here.
# Otherwise, the function will use all rounds and ROIs from the config.

specific_code_fov_pairs = [(code_val, fov_val) for code_val in args.codes for fov_val in args.fovs]  # or [(1,2), (2,3), ...] for spesifc rounds/rois alignment

volumetric_alignment(
    args=args,
    code_fov_pairs=specific_code_fov_pairs,
    parallel_processes=parallelization,
    method=alignment_method,
    bg_sub=background_subtraction
)

# Note: Always monitor the logs or console output to ensure the alignment process 
# is proceeding without errors.





