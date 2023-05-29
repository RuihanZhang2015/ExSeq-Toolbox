"""
Sets up the project parameters. 
"""
import os
import json
import pickle
import pandas as pd

pd.set_option("display.expand_frame_repr", False)
from nd2reader import ND2Reader
from exm.utils import chmod
from exm.io import createFolderStruc



class Args:
    def __init__(self):
        pass

    def set_params(self,
                project_path = '',
                codes = list(range(7)),
                fovs = None,
                ref_code = 0,
                thresholds = [200,300,300,200],
                spacing = [1.625,1.625,4.0],
                gene_digit_csv = 'gene_list.csv', #'/mp/nas3/ruihan/20230308_celegans/code0/gene_list.csv'
                permission = False,
                Create_directroy_Struc = False,
                args_file_name = None,
                ):
        
        r"""Sets parameters for running alignment code.
        :param str project_path: path to project data.
        :param list codes: a list of integers, where each integer represents a code.
        :param list fovs: a list of integers, where each integer represents a field of view.
        :param int ref_code: integer that specifies which code is the reference round.
        :param list thresholds: list of integers, where each integer is a threshold for the code of the same index. Should be the same length as the codes parameter.
        :param str align_z_init: path to ``.pkl`` file that has initial :math:`z`-alignment positions.
        :param bool Create_directroy_Struc: If ``True``, Create the working folders stucture for the project path. Default: ``False``
        :param bool permission: Give other users the permission to read and write on the generated files (linux and macOS only). Default: ``False``
        :param str args_file_name: Pickle file name to store the project Args. Default: ``args.pkl``
        """
        self.project_path = project_path
        self.codes = codes
        self.thresholds = thresholds
        self.spacing = spacing
        self.permission = permission
        self.ref_code = ref_code
        
        # Housekeeping
        self.code2num = {'a':'0','c':'1','g':'2','t':'3'}
        self.colors = ['red','yellow','green','blue']
        self.colorscales = ['Reds','Oranges','Greens','Blues']
        self.channel_names = ['640','594','561','488','405']
        
        # Input ND2 path
        self.nd2_path = os.path.join(
            self.project_path, "code{}/Channel{} SD_Seq000{}.nd2"
        )
        if not fovs and "fovs" not in dir(self):
            self.fovs = list(
                ND2Reader(self.nd2_path.format(self.ref_code, "405", 4)).metadata[
                    "fields_of_view"
                ]
            )
        else:
            self.fovs = fovs

        # Output h5 path
        self.processed_path = os.path.join(self.project_path, "processed_ruihan")
        self.h5_path = os.path.join(self.processed_path, "code{}/{}.h5")
        self.tform_path = os.path.join(self.processed_path, "code{}/tforms/{}.txt")

        # Housekeeping
        self.code2num = {"a": "0", "c": "1", "g": "2", "t": "3"}
        self.colors = ["red", "yellow", "green", "blue"]
        self.colorscales = ["Reds", "Oranges", "Greens", "Blues"]
        self.channel_names = ["640", "594", "561", "488", "405"]

        self.work_path = self.project_path + "puncta/"

        self.gene_digit_csv = gene_digit_csv

        if Create_directroy_Struc:
            createFolderStruc(project_path, codes)

        if args_file_name == None:
            args_file_name = 'args.pkl'

        with open(os.path.join(self.project_path, args_file_name), "wb") as f:
            pickle.dump(self.__dict__, f)

        if permission:
            chmod(self.project_path)


    # load parameters from a pre-set .pkl file
    def load_params(self, param_path):
        r"""Loads and sets attributes from a .pkl file.

        :param str param_path: ``.pkl`` file path.
        """

        with open(os.path.abspath(param_path), "rb") as f:
            self.__dict__.update(pickle.load(f))


    def print(self):
        r"""Prints all attributes.
        """
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")


    def list_output_directory(self):
        r"""Lists the files in the output directory."""
        for root, dirs, files in os.walk(self.processed_path):
            level = root.replace(self.processed_path, "").count(os.sep)
            indent = " " * 4 * (level)
            print("{}{}/".format(indent, os.path.basename(root)))
            subindent = " " * 4 * (level + 1)
            for f in files:
                print("{}{}".format(subindent, f))
                

    def progress(self):
        """visualize_progress ``(self)``"""
        import seaborn as sns
        import numpy as np
        import h5py
        import matplotlib.pyplot as plt

        result = np.zeros((len(self.fovs), len(self.codes)))
        annot = np.asarray(
            [["{},{}".format(fov, code) for code in self.codes] for fov in self.fovs]
        )
        for fov in self.fovs:
            for code_index, code in enumerate(self.codes):

                if os.path.exists(self.h5_path.format(code, fov)):
                    result[fov, code_index] = 1
                else:
                    continue

                if os.path.exists(
                    self.work_path + "/fov{}/result_code{}.pkl".format(fov, code)
                ):
                    result[fov, code_index] = 4
                    continue
                        
                if os.path.exists(self.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code)):
                    result[fov,code_index] = 3
                    continue

                try:
                    with h5py.File(self.h5_path.format(code, fov), "r+") as f:
                        if set(f.keys()) == set(self.channel_names):
                            result[fov, code_index] = 2
                except:
                    pass

        fig, ax = plt.subplots(figsize=(7, 20))
        ax = sns.heatmap(result, annot=annot, fmt="", vmin=0, vmax=4)
        plt.show()
        print(
            "1: 405 done, 2: all channels done, 3:puncta extracted 4:channel consolidated"
        )
