from exm.utils.log import configure_logger
from exm.args.args import Args

# Configure logger for ExSeq-Toolbox
logger = configure_logger('ExSeq-Toolbox')

# Initialize the configuration object.
args = Args()

# ================== Mandatory Configuration ==================
# The absolute path to the raw data directory. Update this path accordingly.
raw_data_directory = '/path/to/your/raw_data_directory/'

# ================== Required Raw Data Directory Structure ==================
# The ExSeq-Toolbox currently assumes the following directory structure:
#
# raw_data_directory/
# ├── code0/
# │   ├── Channel405 SD_Seq0004.nd2
# │   ├── Channel488 SD_Seq0003.nd2
# │   ├── Channel561 SD_Seq0002.nd2
# │   ├── Channel594 SD_Seq0001.nd2
# │   ├── Channel604 SD_Seq0000.nd2
# ├── code1/
# │   ├── Channel405 SD_Seq0004.nd2
# │   ├── ...
# ├── ...
# 
# Ensure that your raw data adheres to this directory structure before running the package.

# ================== Optional Configuration ==================

# List of integers representing specific codes. Default: integers 0-6.
codes_list = list(range(7))

# Optional: List of integers specifying which fields of view to analyze.
# If not provided, all available FOVs will be analyzed.
fov_list = list(range(12))  # Example values

# The absolute path to the processed data directory.
# By default, a 'processed_data' subdirectory inside the raw_data_path is used.
processed_data_directory = '/path/to/processed/data_directory/'

# Spacing between pixels in the format [Z, Y, X].
pixel_spacing = [0.4, 1.625, 1.625]

# Set the names of channels in the ND2 file.
channel_names_list = ['640', '594', '561', '488', '405']

# Specifies which code to use as the reference round. Default: 0.
reference_code = 0

# Specifies which channel to use as the reference for alignment.
reference_channel = '405'

# Absolute path to the CSV file containing the gene list.
gene_digit_file = './gene_list.csv'

# If set to True, changes permission of the raw_data_path to allow other users to read and write on the generated files (Only for Linux and MacOS users).
permissions_flag = False

# If set to True, creates the directory structure in the specified project path.
create_directory_structure_flag = True

# The name of the JSON file where the project arguments will be stored.
args_file = "ExSeq_toolbox_args"

# Set the parameters using the set_params method
args.set_params(
    raw_data_path=raw_data_directory,
    processed_data_path=processed_data_directory,
    codes=codes_list,
    fovs=fov_list,
    spacing=pixel_spacing,
    channel_names=channel_names_list,
    ref_code=reference_code,
    ref_channel=reference_channel,
    gene_digit_csv=gene_digit_file,
    permission=permissions_flag,
    create_directroy_structure=create_directory_structure_flag,
    args_file_name=args_file
)

# Note: Always ensure that the paths and other configuration parameters are correct before running the script.

