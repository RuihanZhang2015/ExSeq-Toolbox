from nd2reader import ND2Reader
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
import seaborn as sns
from numbers_parser import Document
import collections
import os
import pickle

class Args():
    
    def __init__(self,
                project_path = '/mp/nas3/ruihan/20220916_zebrafish/',
                codes = [0,1,2,5],
                fovs = None,
                ref_code = 0,
                thresholds = None,
                ):
        
        if os.path.isfile(project_path + 'args2.pkl'):
            with open(project_path + 'args2.pkl','rb') as f:
                self.__dict__.update(pickle.load(f))
        else:
            self.project_path = project_path
        # self.project_path = project_path

        # Input ND2 path
        if not hasattr(self,'nd2_path'):
            self.nd2_path = self.project_path + 'code{}/Channel{} SD_Seq000{}.nd2'

        # Output h5 path
        if not hasattr(self,'h5_path'):
            self.h5_path = self.project_path + 'processed/code{}/{}.h5'
            self.tform_path = self.project_path + 'processed/code{}/tforms/{}.txt'
        
        # Cropped temporary h5 path
        if not hasattr(self,'h5_path_cropped'):
            self.h5_path_cropped = self.project_path + 'processed/code{}/{}_cropped.h5'
        
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

        from exm.align.starting import starting
        self.starting = starting
        with open(self.project_path + 'args2.pkl','wb') as f:
            pickle.dump(self.__dict__,f)
        self.chmod()
    
    def send_slack(self,message):
        import os
        os.system("curl -X POST -H \'Content-type: application/json\' --data \'{\"text\":\" + 'amama'+   '\"}\' https://hooks.slack.com/services/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")

        
    def print(self):
        for attr in dir(self):
            # print(attr)
            if not attr.startswith('__'):
                print(attr,getattr(self,attr))

    def tree(self):
        import os
        startpath = self.project_path + '/processed/'
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))
      
    def chmod(self):
        import os
        os.system('chmod 777 -R {}'.format(self.project_path))

    def retrieve_all_puncta(self,fov):
        with open(self.project_path + 'processed/fov{}/result.pkl'.format(fov), 'rb') as f:
            return pickle.load(f)
    
    def retrieve_one_puncta(self,fov,puncta_index):
        return self.retrieve_all_puncta(fov)[puncta_index]


    def retrieve_img(self,fov,code,c,ROI_min,ROI_max):

        import h5py
        import numpy as np
        if ROI_min != ROI_max:
            print('use middle z slices')
            zz = int((ROI_min[0]+ROI_max[0])//2)

        with h5py.File(self.h5_path.format(code,fov), "r") as f:
            im = f[self.channel_names[c]][zz, max(0,ROI_min[1]):min(2048,ROI_max[1]), max(0,ROI_min[2]):min(2048,ROI_max[2])]
            im = np.squeeze(im)

        return im
        
    def retrieve_vol(self,fov,code,c,ROI_min,ROI_max):
        
        import h5py
        with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
            vol = f[self.args.channel_names[c]][max(0,ROI_min[0]):ROI_max[0],max(0,ROI_min[1]):min(2048,ROI_max[1]),max(0,ROI_min[2]):min(2048,ROI_max[2])]    
        return vol
        
    # def retrieve_result(self,fov):
    #     with open(self.args.work_path + '/fov{}/result.pkl'.format(fov), 'rb') as f:
    #         return pickle.load(f)
        
    # def retrieve_puncta(self,fov,puncta_index):
    #     return self.retrieve_result(fov)[puncta_index]

    # def retrieve_complete(self,fov):
    #     with open(self.args.work_path+'/fov{}/complete.pkl'.format(fov),'rb') as f:
    #         return pickle.load(f)
        
    # def retrieve_coordinate(self):
    #     with open(self.args.layout_file,encoding='utf-16') as f:
    #         contents = f.read()

    #         contents = contents.split('\n')
    #         contents = [line for line in contents if line and line[0] == '#' and 'SD' not in line]
    #         contents = [line.split('\t')[1:3] for line in contents]

    #         coordinate = [[float(x) for x in line] for line in contents ]
    #         coordinate = np.asarray(coordinate)

    #         # print('oooold',coordinate[:10])

    #         coordinate[:,0] = max(coordinate[:,0]) - coordinate[:,0]
    #         coordinate[:,1] -= min(coordinate[:,1])
    #         coordinate = np.round(np.asarray(coordinate/0.1625/(0.90*2048))).astype(int)
    #         return coordinate

    # def retrieve_coordinate2(self):
        import xml.etree.ElementTree 
        def get_offsets(filename= "/mp/nas3/fixstars/yves/zebrafish_data/20221025/code2/stitched_raw_small.xml"):
            tree = xml.etree.ElementTree.parse(filename)
            root = tree.getroot()
            vtrans = list()
            for registration_tag in root.findall('./ViewRegistrations/ViewRegistration'):
                tot_mat = np.eye(4, 4)
                for view_transform in registration_tag.findall('ViewTransform'):
                    affine_transform = view_transform.find('affine')
                    mat = np.array([float(a) for a in affine_transform.text.split(" ")] + [0, 0, 0, 1]).reshape((4, 4))
                    tot_mat = np.matmul(tot_mat, mat)
                vtrans.append(tot_mat)
            def transform_to_translate(m):
                m[0, :] = m[0, :] / m[0][0]
                m[1, :] = m[1, :] / m[1][1]
                m[2, :] = m[2, :] / m[2][2]
                return m[:-1, -1]

            trans = [transform_to_translate(vt).astype(np.int64) for vt in vtrans]
            return np.stack(trans)

        coordinate = get_offsets()
        coordinate = np.asarray(coordinate)

        coordinate[:,0] = max(coordinate[:,0]) - coordinate[:,0]
        coordinate[:,1] -= min(coordinate[:,1])
        coordinate[:,2] -= min(coordinate[:,2])
        # coordinate[:,:2] = np.asarray(coordinate[:,:2]/0.1625)
        # coordinate[:,2] = np.asarray(coordinate[:,2]/0.4)
        return coordinate

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
   