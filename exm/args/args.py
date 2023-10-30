"""
Sets up the project parameters. 
"""
import os
import json
import pathlib
from nd2reader import ND2Reader
from typing import List, Optional
from exm.utils.log import configure_logger
logger = configure_logger('ExSeq-Toolbox')


class Args:
    r"""
    A class used to represent the configuration for ExSeq-Toolbox.
    """
    
    def __init__(self):
        pass

    def __str__(self):
        r"""Returns a string representation of the Args object."""
        return str(self.__dict__)
    
    def __repr__(self):
        r"""Returns a string that reproduces the Args object when fed to eval()."""
        return self.__str__()

    def set_params(self,
                raw_data_path: str,
                processed_data_path: Optional[str] = None,
                codes: List[int] = list(range(7)),
                fovs: Optional[List[int]] = None,
                spacing: List[float] = [4.0,1.625,1.625],
                channel_names: List[str] = ['640','594','561','488','405'],
                ref_code: int = 0,
                ref_channel: str = '405',
                gene_digit_csv: str = './gene_list.csv',
                permission: Optional[bool] = False,
                create_directroy_structure: Optional[bool] = True,
                args_file_name: Optional[str] = 'exseq_toolbox_args',
                ) -> None:
        r"""
        Sets parameters for running ExSeq ToolBox.

        :param raw_data_path: The absolute path to the project's raw data directory (.nd2 files). There is no default value, this must be provided.
        :type raw_data_path: str
        :param processed_data_path: The absolute path to the processed data directory. Default is a 'processed_data' subdirectory inside the raw_data_path.
        :type processed_data_path: Optional[str]
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
        self.codes = codes
        self.channel_names = channel_names
        self.spacing = spacing
        self.permission = permission
        self.ref_code = ref_code
        self.ref_channel = ref_channel
        self.gene_digit_csv = gene_digit_csv
        
        # Housekeeping
        self.code2num = {'a':'0','c':'1','g':'2','t':'3'}
        self.colors = ['red','yellow','green','blue']
        self.colorscales = ['Reds','Oranges','Greens','Blues']
        self.thresholds = [200,300,300,200]
        
        # Input ND2 path
        self.nd2_path = os.path.join(
            self.raw_data_path, "code{}/Channel{} SD_Seq000{}.nd2"
        )

        if processed_data_path is not None:
            self.processed_data_path = os.path.abspath(processed_data_path)
        else:
            self.processed_data_path = os.path.join(self.raw_data_path, "processed_data")

        self.h5_path = os.path.join(self.processed_data_path, "code{}/{}.h5")
        self.tform_path = os.path.join(self.processed_data_path, "code{}/tforms/{}")
        self.puncta_path = os.path.join(self.processed_data_path, "puncta/")

        if not fovs and "fovs" not in dir(self):
            self.fovs = list(
                ND2Reader(self.nd2_path.format(self.ref_code, self.ref_channel, 4)).metadata[
                    "fields_of_view"
                ]
            )
        else:
            self.fovs = fovs

        if create_directroy_structure is not None:
            self.create_directroy_structure()

        if permission:
            self.set_permissions()

        self.save_params(args_file_name)


    def save_params(self, args_file_name):
        r"""Saves the parameters to a .json file.

        :param args_file_name: Name of the parameters file.
        :type args_file_name: str
        """
        try:
            with open(os.path.join(self.processed_data_path , args_file_name + '.json'), "w") as f:
                json.dump(self.__dict__, f)
        except Exception as e:
            logger.error(f"Failed to save configuration. Error: {e}")
            raise
    
    def create_directroy_structure(self):
        r"""Creates the directory structure in the specified project path."""
        from exm.io import create_folder_structure
        try:
            create_folder_structure(str(self.processed_data_path),self.fovs, self.codes)
        except Exception as e:
            logger.error(f"Failed to create directory structure. Error: {e}")
            raise


    def set_permissions(self):
        r"""Changes permission of the processed_data_path to allow other users to read and write on the generated files."""
        from exm.utils.utils import chmod
        try:
            chmod(pathlib.Path(self.processed_data_path))
        except Exception as e:
            logger.error(f"Failed to set permissions. Error: {e}")
            raise

    def load_params(self, param_path: str) -> None:
        r"""Loads and sets attributes from a .json file.

        :param param_path: ``.json`` file path.
        :type param_path: str
        """
        try:
            param_path = os.path.abspath(param_path)
            with open(param_path, "r") as f:
                self.__dict__.update(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load parameters. Error: {e}")
            raise


    def print(self) -> None:
        r"""Prints all attributes of the Args object."""
        try:
            for key, value in self.__dict__.items():
                print(f"{key}: {value}")
        except Exception as e:
            logger.error(f"Failed to print parameters. Error: {e}")
            raise
