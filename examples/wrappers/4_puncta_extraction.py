from exm.args.args import Args
from exm.puncta.extract import extract
from exm.puncta.consolidate import consolidate_channels, consolidate_codes

# Step 1: Load Configuration Settings
# ====================================

# Initialize the Args object for configuration.
args = Args()

# Provide the path to the configuration file.
args_file_path = '/path/to/your/parameter/file.json'
# Load the configuration settings from the file.
args.load_params(args_file_path)

# Ensure the configuration file path is correct to avoid inconsistencies during processing.

# Step 2: Puncta Extraction
# ====================================

# Define the list of Code, FOV pairs for extraction.
code_fov_pairs_to_extract = [(code_val, fov_val) for code_val in args.codes for fov_val in args.fovs]

# Parameters for extraction.
use_gpu_setting = True  # Whether to use GPU for extraction. Set to False for CPU.
num_gpus = 3  # Number of GPUs to use if use_gpu_setting is True.
num_cpus = 3  # Number of CPUs to use if use_gpu_setting is False.

extract(args=args,
        code_fov_pairs=code_fov_pairs_to_extract,
        use_gpu=use_gpu_setting,
        num_gpu=num_gpus,
        num_cpu=num_cpus)

# Step 3: Channel Consolidation
# ====================================

# Parameters for channel consolidation.
num_cpus_for_channel = 4  # Number of CPUs for channel consolidation.

consolidate_channels(args=args,
                     code_fov_pairs=code_fov_pairs_to_extract,
                     num_cpu=num_cpus_for_channel)

# Step 4: Code Consolidation
# ====================================

# List of FOVs for code consolidation.
fov_list_to_consolidate = list(args.fovs)
num_cpus_for_code = 4  # Number of CPUs for code consolidation.

consolidate_codes(args=args, fov_list=fov_list_to_consolidate, num_cpu=num_cpus_for_code)

# Monitor logs or console output for any errors during processing.
