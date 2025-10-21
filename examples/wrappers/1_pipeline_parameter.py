from exm.utils.log import configure_logger
from exm.args.args import Args

# Configure logger for ExSeq-Toolbox
logger = configure_logger('ExSeq-Toolbox')

# Initialize the configuration object
args = Args()

# ================== Mandatory Configuration ==================
# The absolute path to the raw data directory. Update this path accordingly.
params = {}
params["raw_data_path"] = '/path/to/your/raw_data_directory/'

# ================== Processing Parameters ==================
# Memory and performance optimization
params["chunk_size"] = 150  # Adjust based on your system memory (default: 100)
params["parallel_processes"] = 4  # Auto-detected if not specified
params["use_gpu_processing"] = True  # Enable GPU if available
params["gpu_memory_fraction"] = 0.8  # Use 80% of GPU memory
params["auto_cleanup_memory"] = True  # Automatic memory cleanup

# Puncta extraction parameters (previously hardcoded)
params["puncta_thresholds"] = [200, 300, 300, 200]  # Custom thresholds per channel
params["puncta_min_distance"] = 7  # Minimum distance between puncta
params["puncta_gaussian_sigma"] = 1.0  # Gaussian filter sigma
params["puncta_exclude_border"] = False  # Exclude border puncta
params["consolidation_distance_threshold"] = 8.0  # Distance for consolidation

# Alignment parameters (previously hardcoded)
params["alignment_downsample_factors"] = (2, 4, 4)  # Downsampling factors
params["alignment_low_percentile"] = 1.0  # Intensity normalization
params["alignment_high_percentile"] = 99.0

# System parameters
params["permission_mode"] = 0o777  # Permission mode for created files

# ================== Required Raw Data Directory Structure ==================
# The ExSeq-Toolbox currently assumes the following directory structure:
#
# raw_data_directory/
# ├── code0/
# │   ├── raw_fov0.h5
# │   ├── raw_fov1.h5
# │   ├── raw_fov2.h5
# │   ├── raw_fov3.h5
# │   ├── raw_fov4.h5
# ├── code1/
# │   ├── raw_fov0.h5
# │   ├── ...
# ├── ...
#
# Important: Each of the raw_fov{}.h5 assumed to contain different datasets for each channel. The dataset name needs to be the channel wavelength, e.g., '640', '594', '561', '488', '405'.
#
# Ensure that your raw data adheres to this directory structure before running the package.

# ================== Optional Configuration ==================

# Optional: Number of imaging rounds in targeted sequencing data.
params["codes"] = list(range(7))  # Default value: 7

# Optional: List of integers specifying which fields of view to analyze.
# If not provided, all available FOVs in the raw_data_directory will be analyzed.

# Uncomment to set the number of FOVs and their list explicitly.
# number_of_fovs = 12
# fov_list = list(range(number_of_fovs))
# params["fovs"] = fov_list

# Optional: The absolute path to the processed data directory.
# processed_data_directory = '/path/to/processed/data_directory/'
# params["processed_data_path"] = processed_data_directory

# Optional: Directory name to store puncta analysis results.
# puncta_dir_name = "puncta/"
# params["puncta_dir_name"] = puncta_dir_name

# Optional: Spacing between pixels in the format [Z, Y, X].
params["spacing"] = [0.4, 1.625, 1.625]  # Default value: [0.4, 1.625, 1.625]

# Optional: Set the names of channels in the ND2 file.
params["channel_names"] = ['640', '594', '561', '488', '405']  # Default names

# Optional: Specifies which code to use as the reference round.
params["ref_code"] = 0  # Default value: 0

# Optional: Specifies which channel to use as the reference for alignment.
params["ref_channel"] = '405'  # Default value: '405'

# Optional: Absolute path to the CSV file containing the gene list.
gene_digit_file = './gene_list.csv'
params["gene_digit_csv"] = gene_digit_file

# Optional: Changes permission of the raw_data_path for other users (Linux/Mac).
permissions_flag = False
params["permission"] = permissions_flag

# Optional: Creates the directory structure in the specified project path.
create_directory_structure_flag = True
params["create_directroy_structure"] = create_directory_structure_flag

# Optional: The name of the JSON file where the project arguments will be stored.
args_file = "ExSeq_toolbox_args"
params["args_file_name"] = args_file

# Call enhanced set_params with all parameters
args.set_params(**params)

# ================== New Enhanced Features ==================

# Get processing recommendations based on your system
recommendations = args.get_processing_recommendations()
logger.info("Processing recommendations for your system:")
for key, value in recommendations.items():
    logger.info(f"  {key}: {value}")

# Save configuration in YAML format for easy editing and sharing
yaml_config_path = args.processed_data_path + "/config.yaml"
args.save_config_yaml(yaml_config_path)
logger.info(f"Configuration saved to {yaml_config_path}")

# Get memory configuration object
memory_config = args.get_memory_config()
if memory_config:
    memory_info = memory_config.get_memory_info()
    logger.info(f"Memory configuration: {memory_info}")

# ================== Configuration Loading Example ==================
# You can also load configuration from a YAML file:
# args.load_config_yaml("examples/config_examples/high_memory_config.yaml")
# args.load_config_yaml("examples/config_examples/low_memory_config.yaml")

logger.info("Enhanced configuration completed successfully!")
logger.info(f"Using chunk size: {args.chunk_size}")
logger.info(f"Parallel processes: {args.parallel_processes}")
logger.info(f"GPU processing enabled: {args.use_gpu_processing}")
logger.info(f"Auto memory cleanup: {args.auto_cleanup_memory}")

# Note: Configuration parameters are now fully customizable and hardware-aware.
# Check the generated config.yaml file to see all available options.