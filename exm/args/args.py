"""
Sets up the project parameters. 
"""
import os
import json
import pickle
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

from exm.utils import chmod
from exm.io import createFolderStruc


class Args():
    
    def __init__(self):
        pass

    def set_params(self,
                project_path = '',
                codes = list(range(7)),
                fovs = list(range(3)),
                ref_code = 0,
                thresholds = [200,300,300,200],
                align_z_init=None,
                spacing = [1.625,1.625,4.0],
                Create_directroy_Struc = False,
                permission = False,
                ):
        
        r"""Sets parameters for running alignment code. 
        Args:
            project_path (str): path to project data.
            codes (list): a list of integers, where each integer represents a code. 
            fovs (list): a list of integers, where each integer represents a field of view.
            ref_code (int): integer that specifies which code is the reference round. 
            thresholds (list): list of integers, where each integer is a threshold for the code of the same index. Should be the same length as the codes parameter.
            align_z_init (str): path to .pkl file that has initial z-alignment positions. 
            Create_directroy_Struc (bool): If True, Create the working folders stucture for the porject path. Default:False
            permission (bool): Give other users the permission to read and write on the generated files (linux and macOS only). Default:False 
        """
        self.project_path = project_path
        self.codes = codes
        self.fovs = fovs
        self.ref_code = ref_code
        self.thresholds = thresholds
        self.spacing = spacing
        self.permission = permission

        # Input ND2 path
        self.nd2_path = os.path.join(self.project_path,'code{}/Channel{} SD_Seq000{}.nd2')

        # Output h5 path
        self.processed_path =  os.path.join(self.project_path,'processed')
        self.h5_path = os.path.join(self.processed_path,'code{}/{}.h5')
        self.tform_path = os.path.join(self.processed_path,'code{}/tforms/{}.txt')
        
        # Cropped temporary h5 path
        self.h5_path_cropped = os.path.join(self.processed_path,'code{}/{}_cropped.h5')

        # Housekeeping

        self.code2num = {'a':'0','c':'1','g':'2','t':'3'}
        self.colors = ['red','yellow','green','blue']
        self.colorscales = ['Reds','Oranges','Greens','Blues']
        self.channel_names = ['640','594','561','488','405']

        # # Initilization for alignment parameter 
        if not align_z_init:
            self.align_z_init = align_z_init
        else:
            with open(align_z_init) as f:
                self.align_z_init = json.load(f)

        self.work_path = self.project_path + 'puncta/'
        
        if Create_directroy_Struc:
            createFolderStruc(project_path,codes)

        with open(os.path.join(self.project_path,'args.pkl'),'wb') as f:
            pickle.dump(self.__dict__,f)

        if permission:             
            chmod(os.path.join(self.project_path,'args.pkl'))
        

    # load parameters from a pre-set .pkl file
    def load_params(self, param_path):
        r"""Loads and sets attributes from a .pkl file. 
        Args:
            param_path (str): .pkl file path.
        """
        
        with open(os.path.abspath(param_path),'rb') as f:
            self.__dict__.update(pickle.load(f))

        
    def print(self):
        r"""Prints all attributes.
        """
        for attr in dir(self):
            # print(attr)
            if not attr.startswith('__'):
                print(attr,getattr(self,attr))


    def tree(self):
        r"""Lists the files in the output directory.
        """
        for root, dirs, files in os.walk(self.processed_path):
            level = root.replace(self.processed_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))
      
        