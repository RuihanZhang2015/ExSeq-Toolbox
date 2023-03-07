from nd2reader import ND2Reader
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import seaborn as sns
from numbers_parser import Document
import collections
import os
import pickle

class Args():
    
    def __init__(self):
        pass

    def set_params(self,
                project_path = '',
                codes = [0,1,2,3],
                fovs = None,
                ref_code = 0,
                thresholds = None,
                align_init=None,
                ):
        
        if os.path.isfile(os.path.join(project_path,'args.pkl')):
            with open(os.path.join(project_path,'args.pkl'),'rb') as f:
                self.__dict__.update(pickle.load(f))
        else:
            self.project_path = project_path

        # Input ND2 path
        if not hasattr(self,'nd2_path'):
            self.nd2_path = os.path.join(self.project_path,'code{}/Channel{} SD_Seq000{}.nd2')

        # Output h5 path
        if not hasattr(self,'h5_path'):
            self.h5_path = os.path.join(self.project_path,'processed/code{}/{}.h5')
            self.tform_path = os.path.join(self.project_path,'processed/code{}/tforms/{}.txt')
        
        # Cropped temporary h5 path
        if not hasattr(self,'h5_path_cropped'):
            self.h5_path_cropped = os.path.join(self.project_path,'processed/code{}/{}_cropped.h5')
        
        # Codes and fovs
        if not ref_code and not hasattr(self,'ref_code'):
            self.ref_code = 0

        if not codes and not hasattr(self,'codes'):
            self.codes = range(7)
        
        if not fovs and not hasattr(self,'fovs'): 
            self.fovs = list(ND2Reader(self.nd2_path.format(self.ref_code,'405',4)).metadata['fields_of_view'])

        # Housekeeping
        if not hasattr(self,'colors'):
            self.code2num = {'a':'0','c':'1','g':'2','t':'3'}
            self.colors = ['red','yellow','green','blue']
            self.colorscales = ['Reds','Oranges','Greens','Blues']
            self.channel_names = ['640','594','561','488','405']
        
        # Thresholds
        if not thresholds and not hasattr(self,'thresholds'):
            self.thresholds = [200,300,300,200]

        # Initilization for alignment parameter 
        if not align_init and not hasattr(self,'align_init'):
            from exm.args.default_align_init import default_starting
            self.align_init = default_starting
        else:
            self.align_init = align_init

        with open(os.path.join(self.project_path,'args.pkl'),'wb') as f:
            pickle.dump(self.__dict__,f)
        

    # load parameters from a pre-set .pkl file
    def load_params(self,param_path):
        with open(os.path.abspath(param_path),'rb') as f:
            self.__dict__.update(pickle.load(f))


    # TODO decide connection to slack if is needed
    def send_slack(self,message):
        os.system("curl -X POST -H \'Content-type: application/json\' --data \'{\"text\":\" + 'amama'+   '\"}\' https://hooks.slack.com/services/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")

        
    def print(self):
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
      
    # TODO clear, move or use the visualization function in args
    
        # ### Visualization
        # doc = Document(sheet_path)
        # sheets = doc.sheets()
        # tables = sheets[0].tables()
        # data = tables[0].rows(values_only=True)

        # df = pd.DataFrame(data[1:], columns=data[0])

        
        # self.map_gene = collections.defaultdict(list)
        # for i in range(len(df)):
        #     temp = df.loc[i,'Barcode']
        #     temp = ''.join([self.code2num[temp[code]] for code in self.codes])
        #     self.map_gene[temp] = df.loc[i,'Gene']

        # colors = sns.color_palette(None, len(self.map_gene.keys()))
        # self.map_color = {a:b for a,b in zip(self.map_gene.keys(),colors)}
       
        # if not gene_list and not hasattr(self,'gene_list'):
        #     self.gene_list = 'gene_list.numbers'

        # if not layout_file and not hasattr(self,'layout_file'):
        #     self.layout_file = project_path + 'code0/out.csv'
   