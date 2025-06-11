from exm.utils.log import configure_logger
from exm.args.args import Args

# Configure logger for ExSeq-Toolbox
logger = configure_logger('ExSeq-Toolbox')

# Initialize the configuration object.
args = Args()

# ================== Mandatory Configuration ==================
# The absolute path to the raw data directory. Update this path accordingly.
params = {}
params["raw_data_path"] = '/orcd/data/edboyden/001/mansour/exseq_dataset/'

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
#number_of_fovs = 3
#fov_list = list(range(number_of_fovs))
#params["fovs"] = fov_list

# Optional: The absolute path to the processed data directory.
processed_data_directory = '/orcd/data/edboyden/002/davy/ExSeq-Toolbox/processed/'
params["processed_data_path"] = processed_data_directory

# Optional: Directory name to store puncta analysis results.
puncta_dir_name = "/orcd/data/edboyden/002/davy/ExSeq-Toolbox/puncta/"
params["puncta_dir_name"] = puncta_dir_name

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
args_file = "/orcd/data/edboyden/002/davy/ExSeq-Toolbox/ExSeq_toolbox_args"
params["args_file_name"] = args_file

# Call set_params with the parameters
args.set_params(**params)

# Note: Always ensure that the paths and other configuration parameters are correct before running the script.