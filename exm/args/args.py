"""
Sets up the project parameters. 
"""
import os
import pickle
from nd2reader import ND2Reader
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

from exm.utils import chmod


class Args():
    
    def __init__(self):
        pass

    def set_params(self,
                project_path = '',
                codes = list(range(7)),
                fovs = None,
                ref_code = 0,
                thresholds = [200,300,300,200],
                align_z_init=None,
                spacing = [1.625,1.625,4.0],
                ):
        
        r"""Sets parameters for running alignment code. 
        Args:
            project_path (str): path to project data.
            codes (list): a list of integers, where each integer represents a code. 
            fovs (list): a list of integers, where each integer represents a field of view.
            ref_code (int): integer that specifies which code is the reference round. 
            thresholds (list): list of integers, where each integer is a threshold for the code of the same index. Should be the same length as the codes parameter.
            align_z_init (str): path to .pkl file that has initial z-alignment positions. 
        """
        self.project_path = project_path
        self.codes = codes
        self.ref_code = ref_code
        self.thresholds = thresholds
        self.spacing = spacing

        # Input ND2 path
        self.nd2_path = os.path.join(self.project_path,'code{}/Channel{} SD_Seq000{}.nd2')

        # Output h5 path
        self.h5_path = os.path.join(self.project_path,'processed/code{}/{}.h5')
        self.tform_path = os.path.join(self.project_path,'processed/code{}/tforms/{}.txt')
        
        # Cropped temporary h5 path
        self.h5_path_cropped = os.path.join(self.project_path,'processed/code{}/{}_cropped.h5')

        # Nd2 Fovs                  
        if not fovs: 
            self.fovs = list(ND2Reader(self.nd2_path.format(self.ref_code,'405',4)).metadata['fields_of_view'])

        # Housekeeping

        self.code2num = {'a':'0','c':'1','g':'2','t':'3'}
        self.colors = ['red','yellow','green','blue']
        self.colorscales = ['Reds','Oranges','Greens','Blues']
        self.channel_names = ['640','594','561','488','405']

        # align_z_init
        if not align_z_init:
            self.align_z_init = align_z_init
        else:
            with open(align_z_init, 'rb') as f:
                z_init = pickle.load(f)
            self.align_z_init = z_init

        self.work_path = self.project_path + 'puncta/'


        with open(os.path.join(self.project_path,'args.pkl'),'wb') as f:
            pickle.dump(self.__dict__,f)

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
        startpath = os.path.join(self.project_path,'processed/')
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))
      
