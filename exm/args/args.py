"""
Sets up the project parameters with enhanced configurability.
"""
import os
import json
import glob
import pathlib
import yaml
from pathlib import Path
from nd2reader import ND2Reader
from typing import List, Optional, Dict, Any, Union
from exm.utils.log import configure_logger
logger = configure_logger('ExSeq-Toolbox')


class Args:
    r"""
    A class used to represent and manage the configuration for ExSeq-Toolbox.

    Attributes are set using the `set_params` method, and the configuration can be
    saved to or loaded from a JSON file. Directory structures for the project can also be created,
    and file permissions can be modified as needed.

    **Methods:**
        - `set_params`: Sets various parameters for the ExSeq-Toolbox.
        - `save_params`: Saves the current parameters to a JSON file.
        - `load_params`: Loads parameters from a JSON file.
        - `create_directory_structure`: Creates the necessary directory structure for the project.
        - `set_permissions`: Sets file permissions for the project directory.
        - `print`: Prints all current parameters.
    """

    def __init__(self):
        # Enhanced processing parameters
        self.chunk_size = 100
        self.gpu_memory_fraction = 0.8
        self.background_subtraction_radius = 50
        self.auto_cleanup_memory = True
        self.parallel_processes = self._auto_detect_parallel_processes()  # Auto-detect
        self.use_gpu_processing = self._auto_detect_gpu()
        
        # Alignment parameters
        self.alignment_downsample_factors = (2, 4, 4)
        self.alignment_low_percentile = 1.0
        self.alignment_high_percentile = 99.0
        
        # Puncta extraction parameters
        self.puncta_min_distance = 7
        self.puncta_gaussian_sigma = 1.0
        self.puncta_exclude_border = False
        self.consolidation_distance_threshold = 8.0
        
        # Basecalling parameters
        self.hamming_distance_threshold = 2
        
        # System parameters
        self.permission_mode = 0o777
        self.temp_directory = None

    def __str__(self):
        r"""Returns a string representation of the Args object."""
        return str(self.__dict__)

    def __repr__(self):
        r"""Returns a string that reproduces the Args object when fed to eval()."""
        return self.__str__()

    def set_params(self,
                   raw_data_path: str,
                   processed_data_path: Optional[str] = None,
                   puncta_dir_name: Optional[str] = 'puncta/',
                   codes: List[int] = list(range(7)),
                   fovs: Optional[List[int]] = None,
                   spacing: List[float] = [0.4, 1.625, 1.625],
                   channel_names: List[str] = [
                       '640', '594', '561', '488', '405'],
                   ref_code: int = 0,
                   ref_channel: str = '405',
                   gene_digit_csv: str = './gene_list.csv',
                   permission: Optional[bool] = False,
                   create_directroy_structure: Optional[bool] = True,
                   args_file_name: Optional[str] = 'exseq_toolbox_args',
                   # Enhanced parameters
                   chunk_size: Optional[int] = None,
                   gpu_memory_fraction: Optional[float] = None,
                   parallel_processes: Optional[int] = None,
                   use_gpu_processing: Optional[bool] = None,
                   puncta_thresholds: Optional[List[int]] = None,
                   auto_cleanup_memory: Optional[bool] = None,
                   **kwargs) -> None:
        r"""
        Sets parameters for running ExSeq ToolBox.

        :param raw_data_path: The absolute path to the project's raw data directory (.nd2 files). There is no default value, this must be provided.
        :type raw_data_path: str
        :param processed_data_path: The absolute path to the processed data directory. Default is a 'processed_data' subdirectory inside the raw_data_path.
        :type processed_data_path: Optional[str]
        :param puncta_dir_name: The directory name to store the puncta analysis in the processed data directory. Default is a 'puncta' subdirectory inside the processed_data_path.
        :type puncta_dir_name: Optional[str]
        :param codes: A list of integers, each representing a specific code. Default: integers 0-6.
        :type codes: List[int]
        :param fovs: A list of integers, each representing a specific field of view. Default: ``None``.
        :type fovs: Optional[List[int]]
        :param spacing: Spacing between pixels in the format [Z,Y,X]. Default: [4.0, 1.625, 1.625].
        :type spacing: List[float]
        :param channel_names: Names of channels in the ND2 file *in the correct sequence*. Default is ['640','594','561','488','405'].
        :type channel_names: List[str]
        :param ref_code: Specifies which code to use as the reference round. Default: 0.
        :type ref_code: int
        :param ref_channel: Specifies which channel to use as the reference for alignment. Default is '405'.
        :type ref_channel: str
        :param gene_digit_csv:  absolute path of the CSV file containing gene list. Default: './gene_list.csv'.
        :type gene_digit_csv: str
        :param permission: If set to ``True``, changes permission of the raw_data_path to allow other users to read and write on the generated files. Default is ``False``. `Only for Linux and MacOS users`
        :type permission: Optional[bool]
        :param create_directroy_structure: If set to ``True``, creates the directory structure in the specified processed_data_path. Default: ``True``.
        :type create_directroy_structure: Optional[bool]
        :param args_file_name: The name of the JSON file to store the project arguments. Default: 'exseq_toolbox_args'.
        :type args_file_name: Optional[str]
        """

        self.raw_data_path = os.path.abspath(raw_data_path)
        self.puncta_dir_name = puncta_dir_name
        self.codes = codes
        self.channel_names = channel_names
        self.spacing = spacing
        self.permission = permission
        self.ref_code = ref_code
        self.ref_channel = ref_channel
        self.gene_digit_csv = gene_digit_csv

        # Housekeeping
        self.code2num = {'a': '0', 'c': '1', 'g': '2', 't': '3'}
        self.colors = ['red', 'yellow', 'green', 'blue']
        self.colorscales = ['Reds', 'Oranges', 'Greens', 'Blues']
        self.thresholds = [200, 300, 300, 200]


        self.data_path = os.path.join(
            self.raw_data_path, "code{}/raw_fov{}.h5"
        )

        if processed_data_path is not None:
            self.processed_data_path = os.path.abspath(processed_data_path)
        else:
            self.processed_data_path = os.path.join(
                self.raw_data_path, "processed_data")

        self.h5_path = os.path.join(self.processed_data_path, "code{}/{}.h5")
        self.tform_path = os.path.join(
            self.processed_data_path, "code{}/tforms/{}")
        self.puncta_path = os.path.join(
            self.processed_data_path, self.puncta_dir_name)

        if not fovs and "fovs" not in dir(self):
            fovs_num = len(glob.glob(self.data_path.format(0,"*")))
            self.fovs = list(range(fovs_num))
        else:
            self.fovs = fovs

        if create_directroy_structure is not None:
            self.create_directroy_structure()

        if permission:
            self.set_permissions()

        # Set enhanced parameters
        if chunk_size is not None:
            self.chunk_size = chunk_size
        
        if gpu_memory_fraction is not None:
            self.gpu_memory_fraction = gpu_memory_fraction
        
        if parallel_processes is not None:
            self.parallel_processes = parallel_processes
        
        if use_gpu_processing is not None:
            self.use_gpu_processing = use_gpu_processing
        
        if puncta_thresholds is not None:
            self.thresholds = puncta_thresholds
        
        if auto_cleanup_memory is not None:
            self.auto_cleanup_memory = auto_cleanup_memory
        
        # Apply any additional keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Set parameter {key} = {value}")

        self.save_params(args_file_name)

    def save_params(self, args_file_name):
        r"""Saves the parameters to a .json file.

        :param args_file_name: Name of the parameters file.
        :type args_file_name: str
        """
        try:
            with open(os.path.join(self.processed_data_path, args_file_name + '.json'), "w") as f:
                json.dump(self.__dict__, f)
        except Exception as e:
            logger.error(f"Failed to save configuration. Error: {e}")
            raise

    def load_params(self, param_path: str) -> None:
        r"""Loads and sets the configuration parameters from a previously saved .json file.

        :param param_path: The path to the '.json' file containing the serialized parameters.
                           This is typically the 'exseq_toolbox_args.json' file generated by 
                           the `set_params` call, located within the processed data directory.
        :type param_path: str
        """
        try:
            param_path = os.path.abspath(param_path)
            with open(param_path, "r") as f:
                self.__dict__.update(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load parameters. Error: {e}")
            raise

    def create_directroy_structure(self):
        r"""Creates the directory structure in the specified project path."""
        from exm.io import create_folder_structure
        try:
            create_folder_structure(str(self.processed_data_path), str(
                self.puncta_dir_name), self.fovs, self.codes)
        except Exception as e:
            logger.error(f"Failed to create directory structure. Error: {e}")
            raise

    def set_permissions(self):
        r"""Changes permission of the processed_data_path to allow other users to read and write on the generated files."""
        try:
            from exm.utils.utils import chmod
            chmod(pathlib.Path(self.processed_data_path))
        except Exception as e:
            logger.error(f"Failed to set permissions. Error: {e}")
            raise

    def print(self) -> None:
        r"""Prints all attributes of the Args object."""
        try:
            for key, value in self.__dict__.items():
                print(f"{key}: {value}")
        except Exception as e:
            logger.error(f"Failed to print parameters. Error: {e}")
            raise
    
    def _auto_detect_parallel_processes(self) -> int:
        """Auto-detect optimal number of parallel processes."""
        try:
            import multiprocessing
            import psutil
            
            cpu_count = multiprocessing.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Conservative estimate: 1 process per 4GB RAM, max 8 processes
            max_by_memory = max(1, int(memory_gb / 4))
            optimal = min(cpu_count, max_by_memory, 8)
            
            logger.info(f"Auto-detected {optimal} parallel processes (CPU: {cpu_count}, Memory: {memory_gb:.1f}GB)")
            return optimal
            
        except ImportError:
            logger.warning("Could not auto-detect parallel processes, using 1")
            return 1
    
    def _auto_detect_gpu(self) -> bool:
        """Auto-detect GPU availability."""
        try:
            import cupy
            gpu_count = cupy.cuda.runtime.getDeviceCount()
            logger.info(f"GPU detected: {gpu_count} device(s) available")
            return True
        except ImportError:
            logger.info("GPU not available (CuPy not installed)")
            return False
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            return False
    
    def get_memory_config(self):
        """Get memory configuration dictionary."""
        return {
            'chunk_size': self.chunk_size,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'auto_cleanup': self.auto_cleanup_memory
        }
    
    def save_config_yaml(self, filename: str) -> None:
        """Save configuration in YAML format for better readability."""
        config = {
            'data_paths': {
                'raw_data_path': self.raw_data_path,
                'processed_data_path': self.processed_data_path,
                'puncta_dir_name': self.puncta_dir_name,
                'gene_digit_csv': self.gene_digit_csv,
            },
            'experiment': {
                'codes': self.codes,
                'fovs': self.fovs,
                'spacing': self.spacing,
                'channel_names': self.channel_names,
                'ref_code': self.ref_code,
                'ref_channel': self.ref_channel,
            },
            'processing': {
                'chunk_size': self.chunk_size,
                'parallel_processes': self.parallel_processes,
                'use_gpu_processing': self.use_gpu_processing,
                'gpu_memory_fraction': self.gpu_memory_fraction,
                'auto_cleanup_memory': self.auto_cleanup_memory,
            },
            'alignment': {
                'downsample_factors': list(self.alignment_downsample_factors),
                'low_percentile': self.alignment_low_percentile,
                'high_percentile': self.alignment_high_percentile,
            },
            'puncta': {
                'thresholds': self.thresholds,
                'min_distance': self.puncta_min_distance,
                'gaussian_sigma': self.puncta_gaussian_sigma,
                'exclude_border': self.puncta_exclude_border,
                'consolidation_distance_threshold': self.consolidation_distance_threshold,
            },
            'system': {
                'permission': self.permission,
                'permission_mode': self.permission_mode,
            }
        }
        
        try:
            with open(filename, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save YAML configuration: {e}")
            raise
    
    def load_config_yaml(self, filename: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(filename, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply configuration sections
            if 'data_paths' in config:
                for key, value in config['data_paths'].items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            if 'experiment' in config:
                for key, value in config['experiment'].items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            if 'processing' in config:
                for key, value in config['processing'].items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            if 'alignment' in config:
                for key, value in config['alignment'].items():
                    attr_name = f'alignment_{key}'
                    if hasattr(self, attr_name):
                        setattr(self, attr_name, value)
            
            if 'puncta' in config:
                for key, value in config['puncta'].items():
                    if key == 'thresholds':
                        self.thresholds = value
                    else:
                        attr_name = f'puncta_{key}'
                        if hasattr(self, attr_name):
                            setattr(self, attr_name, value)
            
            if 'system' in config:
                for key, value in config['system'].items():
                    if hasattr(self, key):
                        setattr(self, key, value)
            
            logger.info(f"Configuration loaded from {filename}")
            
        except Exception as e:
            logger.error(f"Failed to load YAML configuration: {e}")
            raise
    
    def get_processing_recommendations(self) -> Dict[str, Any]:
        """Get processing recommendations based on current configuration."""
        return {
            'current_chunk_size': self.chunk_size,
            'current_parallel_processes': self.parallel_processes,
            'gpu_enabled': self.use_gpu_processing,
            'auto_cleanup_enabled': self.auto_cleanup_memory,
        }
