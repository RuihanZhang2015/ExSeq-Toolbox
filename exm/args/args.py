from nd2reader import ND2Reader
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import seaborn as sns
# from numbers_parser import Document
import collections
import os
import pickle

class Args():
    
    def __init__(self):
        pass

    def set_params(self,
                project_path = '',
                codes = list(range(7)),
                fovs = None,
                ref_code = 0,
                thresholds = [200,300,300,200],
                align_init=None,
                spacing = [1.625,1.625,4.0],
                ):
        
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

        self.work_path = self.project_path + 'puncta/'
        
        # Initilization for alignment parameter 
        if not align_init:
            from exm.args.default_align_init import default_starting
            self.align_init = default_starting

        with open(os.path.join(self.project_path,'args.pkl'),'wb') as f:
            pickle.dump(self.__dict__,f)
        

    # load parameters from a pre-set .pkl file
    def load_params(self,param_path):
        with open(os.path.abspath(param_path),'rb') as f:
            self.__dict__.update(pickle.load(f))

        
    def print_args(self):
        for attr in dir(self):
            # print(attr)
            if not attr.startswith('__'):
                print(attr,getattr(self,attr))


    def tree(self):
        startpath = os.path.join(self.project_path,'processed/')
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))
      
