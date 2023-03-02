import seaborn as sns
from ..config.utils import load_cfg
# from ..align.build import alignBuild
from numbers_parser import Document
from ..io.io import createFolderStruc
# from ..io.io import nd2ToVol
# from nd2reader import ND2Reader
import collections
import time
import stat

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import pandas as pd
pd.set_option('display.expand_frame_repr', False)

import h5py
import pickle
import cupy as cp
import numpy as np
from tqdm import tqdm


    
class ExSeq():
    
    def __init__(self, args):
        self.args = args
        self.folder_structure()
        
    def folder_structure(self):
        
        '''Create args.work_path'''
        
        if not os.path.exists(self.args.work_path):
            os.makedirs(self.args.work_path)
            
        for fov in self.args.fovs:
            
            if not os.path.exists(self.args.work_path + '/fov{}/'.format(fov)):
                os.makedirs(self.args.work_path + '/fov{}/'.format(fov))
        
        if not os.path.exists(self.args.out_dir):
            print(self.args.out_dir)
            os.makedirs(self.args.out_dir)
            
        for code in self.args.codes:
            
            if not os.path.exists(self.args.out_dir + 'code{}/'.format(code)):
                os.makedirs(self.args.out_dir + 'code{}/'.format(code))
            
            if not os.path.exists(self.args.out_dir + 'code{}/tforms/'.format(code)):
                os.makedirs(self.args.out_dir + 'code{}/tforms/'.format(code))
            
        try:
            os.systems('chmod 777 -R '+ self.args.work_path)
            os.systems('chmod 777 -R '+ self.args.out_path)
    
        except:
            pass
       
    ### =============== Retrieve =====================
    def retrieve_img(self,fov,code,c,ROI_min,ROI_max):
        from exm.puncta.retrieve import retrieve_img
        return retrieve_img(self,fov,code,c,ROI_min,ROI_max)
    
    def retrieve_vol(self,fov,code,c,ROI_min,ROI_max):
        from exm.puncta.retrieve import retrieve_vol
        return retrieve_vol(self,fov,code,c,ROI_min,ROI_max)

    def retrieve_result(self,fov):
        from exm.puncta.retrieve import retrieve_result
        return retrieve_result(self,fov)
    
    def retrieve_puncta(self,fov,puncta_index):
        from exm.puncta.retrieve import retrieve_puncta
        return retrieve_puncta(self,fov,puncta_index)
    
    def retrieve_complete(self,fov):
        from exm.puncta.retrieve import retrieve_complete
        return retrieve_complete(self,fov)
        
    def retrieve_coordinate(self):
        from exm.puncta.retrieve import retrieve_coordinate
        return retrieve_coordinate(self)
        
    def retrieve_coordinate2(self):
        from exm.puncta.retrieve import retrieve_coordinate2
        return retrieve_coordinate2(self)
       
    
    ### =============== Align =====================
    def transform_405_truncated(self,fov_code_pairs):
        from exm.align.align import transform_405_truncated
        transform_405_truncated(self,fov_code_pairs)
        
    def transform_405_full(self,fov_code_pairs):
        from exm.align.align import transform_405_full
        transform_405_full(self,fov_code_pairs)

    def transform_others_full(self,fov_code_pairs,num_cpu):
        from exm.align.align import transform_others_full
        transform_others_full(self,fov_code_pairs,num_cpu)

    def inspect_alignment(self,fov_code_pairs,temp_dir):
        from exm.align.align import inspect_alignment
        inspect_alignment(self,fov_code_pairs,temp_dir)


    ### =============== Consolidate =====================
    def extract(self,fov_code_pairs,use_gpu=False,num_gpu = 3,num_cpu = 3,chunk_size=100):
        from exm.puncta.extract import extract
        return extract(self,fov_code_pairs,use_gpu,num_gpu,num_cpu,chunk_size)
       
    def consolidate_channels_function(self,fov,code):
        from exm.puncta.consolidate import consolidate_channels_function
        consolidate_channels_function(self,fov,code)

    def consolidate_channels(self,fov_code_pairs):
        from exm.puncta.consolidate import consolidate_channels
        consolidate_channels(self,fov_code_pairs)

    def consolidate_codes(self,fovs,codes=range(7)):
        from exm.puncta.consolidate import consolidate_codes
        consolidate_codes(self,fovs,codes=range(7))
    

    ### =============== Prior information Puncta Extraction ==========
    def prior_info_puncta_function(self,tasks_queue,lock_q,num_missed_round,ROI_size,threshold_fraction):
        
        from scipy.ndimage import gaussian_filter
        from skimage.feature import peak_local_max

        while True: # Check for remaining task in the Queue
            try:
                with lock_q:
                    temp_args = tasks_queue.get_nowait()
                    print('Remaining tasks to process : {}'.format(tasks_queue.qsize()))
            except  queue.Empty:
                print("No task left for "+ current_process().name)
                break

            else:
                fov = temp_args
                print(fov)
                coords_total = dict()
                with open(self.args.work_path + '/fov{}/result.pkl'.format(fov), 'rb') as f:
                    fov_puncta_data = pickle.load(f)
                    
                for i,puncta in enumerate(fov_puncta_data):
                    missed_code = []
                    for code in self.args.codes:
                        if "code{}".format(code) not in puncta:
                            missed_code.append(code)
            # show progress
                    if i % 1000 == 0 :
                        print(i,len(fov_puncta_data))
            # condition to limit ROI local search for puncta when many or no missing rounds presents
                    if (len(missed_code) > num_missed_round) or missed_code == []:
                        pass
                    else: 
                        ROI_min = [puncta['position'][0]-ROI_size,puncta['position'][1]-ROI_size,puncta['position'][2]-ROI_size]
                        ROI_max = [puncta['position'][0]+ROI_size,puncta['position'][1]+ROI_size,puncta['position'][2]+ROI_size]

                        for code in missed_code:
                            coords_total["code{}".format(code)] = {}
                            for c in range(4):

                                with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                                    img = f[self.args.channel_names[c]][ROI_min[0]:ROI_max[0],ROI_min[1]:ROI_max[1],ROI_min[2]:ROI_max[2]]
                                    f.close()

                                gaussian_filter(img, 1, output=img, mode='reflect', cval=0)
                                # coords = peak_local_max(img, min_distance = 7, threshold_abs=(args['thresholds'][c])*threshold_fraction,exclude_border=False)
                                coords = peak_local_max(img, min_distance = 7, threshold_abs=130,exclude_border=False)

                                coords = coords + ROI_min
                                coords_total["code{}".format(code)]['c{}'.format(c)] = coords

                            with open(self.args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'rb') as f:
                                coords_org = pickle.load(f)

                            with open(self.args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'wb') as f:
                                for channel in range(4):
                                    coords_org['c{}'.format(channel)] = np.concatenate((coords_org['c{}'.format(channel)],coords_total['code{}'.format(code)]['c{}'.format(channel)]),axis = 0)
                                pickle.dump(coords_org,f)

 
    def prior_info_puncta_function_acceleration(self,tasks_queue,device,lock,lock_q,num_missed_round,ROI_size,threshold_fraction):
        
        from cupyx.scipy.ndimage import gaussian_filter
        from cucim.skimage.feature import peak_local_max

        with cp.cuda.Device(device):
            while True: # Check for remaining task in the Queue
                try:
                    with lock_q:
                        temp_args = tasks_queue.get_nowait()
                        print('Remaining tasks to process : {}'.format(tasks_queue.qsize()))
                except  queue.Empty:
                    print("No task left for "+ current_process().name)
                    break

                else:
                    fov = temp_args

                    # Read consolidate code results.pkl file for each FOV 
                    coords_total = dict()
                    with open(self.args.work_path + '/fov{}/result.pkl'.format(fov), 'rb') as f:
                        fov_puncta_data = pickle.load(f)

                    # Find missing rounds in each dedected puncta     
                    for i,puncta in enumerate(fov_puncta_data):
                            missed_code = []
                            for code in self.args.codes:
                                if "code{}".format(code) not in puncta:
                                    missed_code.append(code)
                    # show progress
                            if i % 1000 == 0 :
                                print(i,len(fov_puncta_data))
                    # condition to limit ROI local search for puncta when many or no missing rounds presents
                            if (len(missed_code) > num_missed_round) or missed_code == []:
                                pass
                            else:
                                print(i,missed_code)
                                #Calculate ROI coordinate around the puncta center
                                ROI_min = [puncta['position'][0]-ROI_size,puncta['position'][1]-ROI_size,puncta['position'][2]-ROI_size]
                                ROI_max = [puncta['position'][0]+ROI_size,puncta['position'][1]+ROI_size,puncta['position'][2]+ROI_size]

                                # ROI serach for puncta in the missing rounds only
                                for code in missed_code:
                                    coords_total["code{}".format(code)] = {}
                                    for c in range(4):

                                        with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                                            img = f[self.args.channel_names[c]][ROI_min[0]:ROI_max[0],ROI_min[1]:ROI_max[1],ROI_min[2]:ROI_max[2]]
                                            f.close()

                                        img = cp.array(img)
                                        gaussian_filter(img, 1, output=img, mode='reflect', cval=0)
                                        # coords = peak_local_max(img, min_distance = 7, threshold_abs= 1(self.args.thresholds[c])*threshold_fraction,exclude_border=False)
                                        coords = peak_local_max(img, min_distance = 7, threshold_abs=120, exclude_border=False)
                                        coords = coords.get() + ROI_min
                                        coords_total["code{}".format(code)]['c{}'.format(c)] = coords
                                        del img
                                        del coords
                                        cp.get_default_memory_pool().free_all_blocks()
                                        cp.get_default_pinned_memory_pool().free_all_blocks()

                                    # Read the avaiable puncta coordinate file 'coords_total_code*.pkl' for that round 
                                    with open(self.args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'rb') as f:
                                        coords_org = pickle.load(f)

                                    # Append the ROI newly dedected puncta per channel to the overall dedected puncta file 'coords_total_code*.pkl'
                                    with open(self.args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'wb') as f:
                                        for channel in range(4):
                                            coords_org['c{}'.format(channel)] = np.concatenate((coords_org['c{}'.format(channel)],coords_total['code{}'.format(code)]['c{}'.format(channel)]),axis = 0)
                                        pickle.dump(coords_org,f)


    def prior_info_puncta_cpu(self,fovs,num_cpu=3,num_missed_round=1,ROI_size=7,threshold_fraction=0.7):
        
        from scipy.ndimage import gaussian_filter
        from skimage.feature import peak_local_max

        # List to hold the child processes.
        child_processes = [] 
        # Queue to hold all the puncta extraction tasks.
        tasks_queue = Queue() 
        # Queue locks to avoid race condition.
        q_lock = multiprocessing.Lock()
        
        start_time = time.time()
        for fov in fovs:
            tasks_queue.put((fov))

        # Excute the extraction tasks on GPU/CPU.

        for w in range(int(num_cpu)):
            p = Process(target=self.prior_info_puncta_function, args=(tasks_queue,q_lock,num_missed_round,ROI_size,threshold_fraction))
            child_processes.append(p)
            p.start()

        for p in child_processes:
            p.join()

        print("find_local_puncta_time",time.time()-start_time,"s")


    def prior_info_puncta_gpu(self,fovs,num_gpu=1,process_per_gpu=1,num_missed_round=1,ROI_size=7,threshold_fraction=0.7):
 
        from cupyx.scipy.ndimage import gaussian_filter
        from cucim.skimage.feature import peak_local_max
        # List to hold the child processes.
        child_processes = [] 
        # Queue to hold all the puncta extraction tasks.
        tasks_queue = Queue() 
        # Queue locks to avoid race condition.
        q_lock = multiprocessing.Lock()
        
        start_time = time.time()
        for fov in fovs:
            tasks_queue.put((fov))

        # Excute the extraction tasks on GPU/CPU.

            # Create locks for each GPU device to avoid OOM situations when many processes simultaneously access the GPU. 
        gpu_locks=[]
        for gpu in range(num_gpu):
            lock = multiprocessing.Lock()
            gpu_locks.append((gpu,lock))

        # Create and start a parallel execution processes based on the number of GPUs and 'process_per_gpu'. 
        process_per_gpu = process_per_gpu * num_gpu
        for gpu_device in gpu_locks:
            for cpu_cores in range(process_per_gpu):
                p = Process(target=self.prior_info_puncta_function_acceleration, args=(tasks_queue,int(gpu_device[0]),gpu_device[1],q_lock,num_missed_round,ROI_size,threshold_fraction))
                child_processes.append(p)
                p.start()

        for p in child_processes:
            p.join()

        print("find_local_puncta_time",time.time()-start_time,"s") 
    
    
    ### ================= Sanity check ====================
    def visualize_progress(self):
        
        import matplotlib.pyplot as plt
        """visualize_progress(self)"""
        
        result = np.zeros((len(self.args.fovs),len(self.args.codes)))
        annot = np.asarray([['{},{}'.format(fov,code) for code in self.args.codes] for fov in self.args.fovs])
        for fov in self.args.fovs:
            for code_index,code in enumerate(self.args.codes):
                
                if os.path.exists(self.args.h5_path.format(code,fov)):
                    result[fov,code_index] = 1
                else:
                    continue
                    
                if os.path.exists(self.args.work_path + '/fov{}/result_code{}.pkl'.format(fov,code)):
                    result[fov,code_index] = 4
                    continue
                    
                if os.path.exists(self.args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code)):
                    
                    result[fov,code_index] = 3
                    continue
                    
                try:
                    with h5py.File(self.args.h5_path.format(code,fov), 'r+') as f:
                        if set(f.keys()) == set(self.args.channel_names):
                            result[fov,code_index] = 2          
                except:
                    pass


        fig, ax = plt.subplots(figsize = (7,20))
        ax = sns.heatmap(result, annot=annot, fmt="",vmin=0, vmax=4)
        plt.show()
        print('1: 405 done, 2: all channels done, 3:puncta extracted 4:channel consolidated')
        

    ### =============== In_Region =======================
    def in_region(self,coord,ROI_min,ROI_max):
        
        """in_region(self,coord,ROI_min,ROI_max)"""

        coord = np.asarray(coord)
        ROI_min = np.asarray(ROI_min)
        ROI_max = np.asarray(ROI_max)

        if np.all(coord>=ROI_min) and np.all(coord<ROI_max):
            return True
        else:
            return False
    

    ### ================== Inspect Raw images =================
    def inspect_stitching(self,code,z):

        def show_fov(fov,code=0,z=0):
            
            with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                img = f[self.args.channel_names[4]][z,:,:]

            fig = plt.figure(figsize = (10,10))
            plt.imshow(img,cmap=plt.get_cmap('gray'),vmax = 600)
            plt.text(0,0,'fov{}'.format(fov),fontsize = 100)
            plt.tight_layout()
            fig.savefig(self.args.work_path + 'max_projection/fov_{}_405_z={}.png'.format(fov,z))
            plt.close(fig)

        if not os.path.exists(self.args.work_path + 'max_projection/'):
            os.makedirs(self.args.work_path + 'max_projection/')

        # ------Get coordinates-----
        # coordinate = pd.read_csv(self.args.layout_file, header = None, sep = ',', dtype = np.float64)
        # coordinate = np.asarray(coordinate)
        # coordinate = coordinate[:,:2]

        # coordinate[:,0] = max(coordinate[:,0]) - coordinate[:,0]
        # coordinate[:,1] -= min(coordinate[:,1])
        # coordinate = np.round(np.asarray(coordinate/0.1625/(0.90*2048))).astype(int)

        coordinate = self.retrieve_coordinate()

        code = 0
        num_row = len(set(coordinate[:,1]))
        num_col = len(set(coordinate[:,0]))
        print('number of rows=',num_row,' number of columns=', num_col)

        # -------Each FOV-------
        for fov in self.args.fovs:
            show_fov(fov, code,z)

        # -------All FOVs-------
        fig,axs = plt.subplots(num_row,num_col,figsize = (3*num_col,3*num_row))
        for fov,(col,row) in enumerate(coordinate):
            img_name = self.args.work_path + 'max_projection/fov_{}_405_z={}.png'.format(fov,z)
            if os.path.exists(img_name):
                img = plt.imread(img_name)
                axs[row,col].imshow(img)


        for row in range(num_row):
            for col in range(num_col):
                axs[row,col].get_xaxis().set_visible(False)
                axs[row,col].get_yaxis().set_visible(False)
                axs[row,col].axis('off')
        plt.tight_layout()     
        plt.savefig(self.args.work_path + 'max_projection/all_405_z={}.png'.format(z))
        print('The stitched image is saved at:', self.args.work_path + 'max_projection/all_405_z={}.png'.format(z))
    
    
    ### ============== Help set the threshold================
    def inspect_raw_plotly(self,fov,code,c,ROI_min,ROI_max,zmax=600):
        from exm.puncta.inspect import inspect_raw_plotly
        inspect_raw_plotly(self,fov,code,c,ROI_min,ROI_max,zmax)
        

    ### ============== Help set the threshold================
    def inspect_raw_matplotlib(self,fov,code,c,ROI_min,ROI_max,vmax = 600):
        from exm.puncta.inspect import inspect_raw_matplotlib
        inspect_raw_matplotlib(self,fov,code,c,ROI_min,ROI_max,vmax = 600)
        
        
    ###============== Help set the threshold================
    def inspect_raw_channels_matplotlib(self,fov,code,ROI_min,ROI_max,vmax = 600):
        from exm.puncta.inspect import inspect_raw_channels_matplotlib
        inspect_raw_channels_matplotlib(self,fov,code,ROI_min,ROI_max,vmax)
       

    ### ============= Inspection Per Channel =======================
    def inspect_localmaximum_plotly(self, fov, code, c, ROI_min, ROI_max):

        from exm.puncta.inspect import inspect_localmaximum_plotly
        inspect_localmaximum_plotly(self, fov, code, c, ROI_min, ROI_max)

    
    ######################################################### 
    #########################################################
    ### ===============Inspect ROI ===================
    def inspect_ROI_matplotlib(self, fov, code, position, centered=40):

        '''
        exseq.inspect_ROI_matplotlib(fov = 21, code = 5, position = [338, 1202, 1383], centered = 40)
        '''
        reference = self.retrieve_result(fov)
        
        fig,axs = plt.subplots(4,10,figsize = (15,7))

        for c in range(4):
            for z_ind,z in enumerate(np.linspace(position[0] - 10,position[0] + 10,10)):
                ROI_min = [int(z),position[1] - centered, position[2] - centered]
                ROI_max = [int(z),position[1] + centered, position[2] + centered]
                img = self.retrieve_img(fov,code,c,ROI_min,ROI_max)
                axs[c,z_ind].imshow(img,cmap=plt.get_cmap(self.args.colorscales[c]),vmax = 150)

        ROI_min = [position[0] - 10,position[1] - centered, position[2] - centered]
        ROI_max = [position[0] + 10,position[1] + centered, position[2] + centered]    
        temp = [x['code{}'.format(code)] for x in reference if 'code{}'.format(code) in x and self.in_region(x['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 

        for c in range(4):
            temp2 = [x['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x]
            for puncta in temp2:
                axs[c,(puncta[0]-position[0]+10)//2].scatter(puncta[1]-position[1]+centered,puncta[2]-position[2]+centered,s = 20)
        
        

        fig.suptitle('fov{} code{}'.format(fov,code))    
        plt.show()
        
    #### ============== Inspect ROI ====================
    def inspect_ROI_plotly(self, fov, ROI_min, ROI_max, codes, c_list=[0,1,2,3],centered=40):

        '''
        exseq.inspect_fov_all(
                fov =
                ,ROI_min =
                ,ROI_max =
                ,codes = 
                ,c_list = [0,1,2,3]
                ,centered = 40
                )
        '''

        spacer = 40

        ROI_center = [(ROI_min[1]+ROI_max[1])//2, (ROI_min[2]+ROI_max[2])//2 ] 

        ROI_min[1] = ROI_center[0] - centered
        ROI_min[2] = ROI_center[1] - centered
        ROI_max[1] = ROI_center[0] + centered
        ROI_max[2] = ROI_center[1] + centered
        print('ROI_min = [{},{},{}]'.format(*ROI_min))
        print('ROI_max = [{},{},{}]'.format(*ROI_max))

        reference = retrieve_result(fov)

        fig = go.Figure()

        for i,code in enumerate(codes):

            ## Surface -------------
            for c in c_list:

                for zz in np.linspace(ROI_min[0],ROI_max[0],7):

                    with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                        im = f[self.args.channel_names[c]][int(zz),ROI_min[1]:ROI_max[1],ROI_min[2]:ROI_max[2]]
                        im = np.squeeze(im)
                    y = list(range(ROI_min[1], ROI_max[1]))
                    x = list(range(ROI_min[2], ROI_max[2]))
                    z = np.ones((ROI_max[1]-ROI_min[1],ROI_max[2]-ROI_min[2])) * ( int(zz)+0.7*c+i*spacer )
                    fig.add_trace(go.Surface(x=x, y=y, z=z,
                            surfacecolor=im,
                            cmin=0, 
                            cmax=500,
                            colorscale=self.args.colorscales[c],
                            showscale=False,
                            opacity = 0.2,
                        ))

            ## Scatter --------------
            temp = [x['code{}'.format(code)] for x in reference if 'code{}'.format(code) in x and self.in_region(x['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 

            for c in c_list:

                temp2 = [x['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x]
                temp2 = np.asarray(temp2)
                if len(temp2)==0:
                    continue
                fig.add_trace(go.Scatter3d(
                        z=temp2[:,0] + i*spacer,
                        y=temp2[:,1],
                        x=temp2[:,2],
                        mode = 'markers',
                        marker = dict(
                            color = self.args.colors[c],
                            size=4,
                        )
                    ))

        # ------------
        fig.add_trace(go.Scatter3d(
                        z= [ROI_min[0], ROI_max[0] + (len(codes)-1)*spacer],
                        y= [(ROI_min[1]+ROI_max[1])/2]*2,
                        x= [(ROI_min[2]+ROI_max[2])/2]*2,
                        mode = 'lines',
                        line = dict(
                            color = 'black',
                            width = 10,
                        )
                    ))        


        # ---------------------
        fig.update_layout(
            title = "Inspect fov{}, code ".format(fov) + 'and '.join([str(x) for x in codes]),
            width = 800,
            height = 800,

            scene=dict(
                aspectmode = 'data',
                xaxis_visible=True,
                yaxis_visible=True, 
                zaxis_visible=True, 
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z" ,
            ))

        fig.show()
        
    ### ==========Inspect puncta matplotlib===================
    def inspect_puncta_matplotlib(self,fov,puncta_index, centered=40):

        '''
        exseq.inspect_puncta_matplotlib(fov,puncta_index = 100, centered = 40)
        '''
        
        puncta = self.retrieve_puncta(fov,puncta_index)
        
        fig,axs = plt.subplots(4,len(self.args.codes),figsize = (15,7))
            
        for code_ind,code in enumerate(self.args.codes):
            
            if 'code{}'.format(code) not in puncta:
                continue
                
            position = puncta['code{}'.format(code)]['position']
            ROI_min = [int(position[0]),position[1] - centered, position[2] - centered]
            ROI_max = [int(position[0]),position[1] + centered, position[2] + centered]
            for c in range(4):
                img = self.retrieve_img(fov,code,c,ROI_min,ROI_max)
                axs[c,code_ind].imshow(img,cmap=plt.get_cmap(self.args.colorscales[c]),vmax = 150)
                axs[c,code_ind].set_title('{0:0.2f}'.format(img[centered,centered]))
            
            axs[puncta['code{}'.format(code)]['color'],code_ind].scatter(centered,centered,c = 'white')

        fig.suptitle('fov{} puncta{}'.format(fov,puncta_index))    
        plt.show() 
        
    ### ========== Inspect puncta plotly==============
    def inspect_puncta_plotly(self, fov, puncta_index,spacer = 40 ):

        '''
        exseq.inspect_puncta(
                fov = 
                ,puncta_index = 
                ,spacer = 40 
                )
        '''

        codes = self.args.codes

        puncta = self.retrieve_puncta(fov,puncta_index)

        
        fig = go.Figure()
        for i, code in enumerate(codes):

            if 'code{}'.format(code) in interest:

                print('code{}'.format(code))
                puncta = reference[puncta_index]['code{}'.format(code)]   
                d0, d1, d2 = puncta['position']
                ROI_min = [d0-10, d1-40, d2-40]
                ROI_max = [d0+10, d1+40, d2+40]

                print('ROI_min = [{},{},{}]'.format(*ROI_min))
                print('ROI_max = [{},{},{}]'.format(*ROI_max))
                # pprint.pprint(puncta)


                c_candidates = []

                ## Surface -------------
                for c in range(4):

                    if 'c{}'.format(c) in puncta:

                        c_candidates.append(c)

                        for zz in np.linspace(ROI_min[0],ROI_max[0],7):

                            with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                                im = f[self.args.channel_names[c]][int(zz),ROI_min[1]:ROI_max[1],ROI_min[2]:ROI_max[2]]
                                im = np.squeeze(im)
                            y = list(range(ROI_min[1], ROI_max[1]))
                            x = list(range(ROI_min[2], ROI_max[2]))
                            z = np.ones((ROI_max[1]-ROI_min[1],ROI_max[2]-ROI_min[2])) * (int(zz)+0.5*c + i* spacer)
                            fig.add_trace(go.Surface(x=x, y=y, z=z,
                                surfacecolor=im,
                                cmin=0, 
                                cmax=500,
                                colorscale=self.args.colorscales[c],
                                showscale=False,
                                opacity = 0.2,
                            ))

                ## Scatter --------------

                temp = [x['code{}'.format(code)] for x in reference if 'code{}'.format(code) in x and self.in_region(x['code{}'.format(code)]['position'], ROI_min,ROI_max) ] 

                for c in c_candidates:

                    fig.add_trace(go.Scatter3d(
                            z = [puncta['c{}'.format(c)]['position'][0]+ i * spacer], 
                            y = [puncta['c{}'.format(c)]['position'][1]],
                            x = [puncta['c{}'.format(c)]['position'][2]],
                            mode = 'markers',
                            marker = dict(
                                color = 'gray',
                                size= 8,
                                symbol = 'circle-open'
                            )
                        ))

                    temp2 = np.asarray([x['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x])

                    if len(temp2) == 0:
                        continue

                    fig.add_trace(go.Scatter3d(
                            z = temp2[:,0] + i* spacer,
                            y = temp2[:,1],
                            x = temp2[:,2],
                            mode = 'markers',
                            marker = dict(
                                color = self.args.colors[c],
                                size=4,
                            )
                        ))

        # ---------------------
        fig.update_layout(
            title="puncta index {}".format(puncta_index),
            width=800,
            height=800,

            scene=dict(
                aspectmode = 'data',
                xaxis_visible = True,
                yaxis_visible = True, 
                zaxis_visible = True, 
                xaxis_title = "X",
                yaxis_title = "Y",
                zaxis_title = "Z" ,
            ))

        fig.show()


        
    ######################################################### 
    #########################################################
    ### ============== inspect_fov_all_to_all ==================
    def inspect_fov_tworounds(self, fov, code1, code2, ROI_min, ROI_max):

        '''
        exseq.inspect_fov_all_to_all(
                fov=
                ,code1=
                ,code2=
                ,ROI_min=
                ,ROI_max=
                )
        '''

        spacer = 100

        with open(self.args.work_path +'/fov{}/result.pkl'.format(fov), 'rb') as f:
            reference = pickle.load(f)
        reference = [ x for x in reference if self.in_region(x['position'], ROI_min,ROI_max) ] 

        fig = go.Figure()

        ## Lines ====================

        temp = [x for x in reference if (code1 in x) and (code2 in x) ]
        for x in temp:
            center1,center2 = x[code1]['position'], x[code2]['position']
            name = x['index']
            fig.add_trace(go.Scatter3d(
                z=[center1[0],center2[0]+spacer],
                y=[center1[1],center2[1]],
                x=[center1[2],center2[2]],
                mode = 'lines',
                name = name,
                line = dict(
                    color = 'gray',
                    # size=4,
                )
            ))
            

        ## Code1  =========================

        temp = [x for x in reference if (code1 in x)]


        ### Centers
        points = [x[code1]['position'] for x in temp]
        points = np.asarray(points)
        texts = [x['index'] for x in temp]
        if len(points)>0:
            fig.add_trace(go.Scatter3d(
                    z=points[:,0],
                    y=points[:,1],
                    x=points[:,2],
                    text = texts,
                    mode = 'markers+text',
                    name = 'consensus',
                    marker = dict(
                        color = 'gray',
                        size=10,
                        opacity = 0.2,
                    )
                ))

        ## Scatters --------------

        for c in range(4):

            points = [x[code1]['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x[code1]]
            points = np.asarray(points)
            if len(points) == 0:
                continue

            fig.add_trace(go.Scatter3d(
                z=points[:,0],
                y=points[:,1],
                x=points[:,2],
                name = 'channels',
                mode = 'markers',
                marker = dict(
                    color = self.args.colors[c],
                    size=4,
                )
            ))

        ## Lines --------------

        for x in temp:
            points = [ x[code1][c]['position'] for c in ['c0','c1','c2','c3'] if c in x[code1] ]

            for i in range(len(points)-1):
                for j in range(i+1,len(points)):

                    fig.add_trace(go.Scatter3d(
                        z = [ points[i][0], points[j][0] ],
                        y = [ points[i][1], points[j][1] ],
                        x = [ points[i][2], points[j][2] ],
                        mode = 'lines',
                        name = 'inter channel',
                        line = dict(
                            color = 'gray',
                            # size=4,
                        )
                    ))   


        ## Code2  =========================


        temp = [x for x in reference if (code2 in x)]

        ### Centers
        points = [x[code2]['position'] for x in temp]
        points = np.asarray(points)
        texts = [x['index'] for x in temp]

        if len(points)>0:
            fig.add_trace(go.Scatter3d(
                z=points[:,0] + spacer,
                y=points[:,1],
                x=points[:,2],
                text = texts,
                mode = 'markers+text',
                name = 'consensus',
                marker = dict(
                    color = 'gray',
                    size=10,
                    opacity = 0.2,
                )
            ))

        ## Scatters --------------

        for c in range(4):

            points = [x[code2]['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x[code2]]
            points = np.asarray(points)
            if len(points) == 0:
                continue
            fig.add_trace(go.Scatter3d(
                z=points[:,0] + spacer,
                y=points[:,1],
                x=points[:,2],
                mode = 'markers',
                name = 'channels',
                marker = dict(
                    color = self.args.colors[c],
                    size=4,
                )
            ))

        ## Lines --------------

        for x in temp:
            points = [ x[code2][c]['position'] for c in ['c0','c1','c2','c3'] if c in x[code2] ]
            for i in range(len(points)-1):
                for j in range(i+1,len(points)):
                    fig.add_trace(go.Scatter3d(
                        z = [ points[i][0]+spacer, points[j][0]+spacer ],
                        y = [ points[i][1], points[j][1] ],
                        x = [ points[i][2], points[j][2] ],
                        mode = 'lines',
                        name = 'inter channel',
                        line = dict(
                            color = 'gray',
                            # size=4,
                        )
                    ))        

        # ---------------------
        fig.update_layout(
            title="My 3D scatter plot",
            width=800,
            height=800,
            showlegend=False,
            scene=dict(
                aspectmode = 'data',
                xaxis_visible=True,
                yaxis_visible=True, 
                zaxis_visible=True, 
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z" ,
            ))

        fig.show()

    ### ============== inspect_fov_all_to_all ==================
    def inspect_fov_allrounds(self, fov, ROI_min, ROI_max):

        '''
        exseq.inspect_fov_all(
                fov=
                ,ROI_min=
                ,ROI_max=
                )
        '''

        spacer = 100

        with open(self.args.work_path +'/fov{}/result.pkl'.format(fov), 'rb') as f:
            reference = pickle.load(f)
        reference = [ x for x in reference if self.in_region(x['position'], ROI_min,ROI_max) ] 
#         print(reference)

        fig = go.Figure()

        ## Lines ====================

        for i1 in range(len(self.args.codes))[:-1]:

            code1 = 'code{}'.format(self.args.codes[i1])
            i2 = i1+1
            code2 = 'code{}'.format(self.args.codes[i2])

            temp = [x for x in reference if (code1 in x) and (code2 in x) ]
            for x in temp:
                    center1,center2 = x[code1]['position'], x[code2]['position']
                    name = x['index']
                    fig.add_trace(go.Scatter3d(
                        z=[center1[0]+i1*spacer,center2[0]+i2*spacer],
                        y=[center1[1],center2[1]],
                        x=[center1[2],center2[2]],
                        mode = 'lines',
                        name = name,
                        line = dict(
                            color = 'gray',
                            # size=4,
                        )
                    ))


        ## Code1  =========================

        for ii,code in enumerate(self.args.codes):

            code1 = 'code{}'.format(code)

            temp = [x for x in reference if (code1 in x)]

            ### Centers
            points = [x[code1]['position'] for x in temp]
            points = np.asarray(points)
            texts = [x['index'] for x in temp]
            if len(points)>0:
                fig.add_trace(go.Scatter3d(
                        z=points[:,0]+ii*spacer,
                        y=points[:,1],
                        x=points[:,2],
                        text = texts,
                        mode = 'markers+text',
                        name = 'consensus',
                        marker = dict(
                            color = 'gray',
                            size=10,
                            opacity = 0.2,
                        )
                    ))

            ## Scatters --------------

            for c in range(4):

                points = [x[code1]['c{}'.format(c)]['position'] for x in temp if 'c{}'.format(c) in x[code1]]
                points = np.asarray(points)
                if len(points) == 0:
                    continue

                fig.add_trace(go.Scatter3d(
                    z=points[:,0]+ii*spacer,
                    y=points[:,1],
                    x=points[:,2],
                    name = 'channels',
                    mode = 'markers',
                    marker = dict(
                        color = self.args.colors[c],
                        size=4,
                    )
                ))

            ## Lines --------------

            for x in temp:
                points = [ x[code1][c]['position'] for c in ['c0','c1','c2','c3'] if c in x[code1] ]

                for i in range(len(points)-1):
                    for j in range(i+1,len(points)):

                        fig.add_trace(go.Scatter3d(
                            z = [ points[i][0]+ii*spacer, points[j][0]+ii*spacer ],
                            y = [ points[i][1], points[j][1] ],
                            x = [ points[i][2], points[j][2] ],
                            mode = 'lines',
                            name = 'inter channel',
                            line = dict(
                                color = 'gray',
                                # size=4,
                            )
                        ))   


        # ---------------------
        fig.update_layout(
            title="My 3D scatter plot",
            width=800,
            height=800,
            showlegend=False,
            scene=dict(
                aspectmode = 'data',
                xaxis_visible=True,
                yaxis_visible=True, 
                zaxis_visible=True, 
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z" ,
            ))

        fig.show()
    

    
    ######################################################### 
    #########################################################    
    def template_matching(self,s):
    
        def match(s,t):
            for i in range(7):
                if s[i] not in ['_',t[i]]:
                    return False
            return True

        out = []
        for template in self.args.map_gene.keys():
            if match(s,template):
                out.append(self.args.map_gene[template])
        return out

    def summary(self):

        coordinate = self.retrieve_coordinate()
        out = collections.defaultdict(list)
        for fov in self.args.fovs:
            result = self.retrieve_result(fov)
            for puncta in result:
                out[puncta['barcode']].append(
                    {
                        'point':[puncta['position'][0],puncta['position'][1]+coordinate[fov,1],puncta['position'][2]+coordinate[fov,0]],
                        'index':'fov:{}<br>index:{}'.format(fov,puncta['index'])
                    }
                )

        realout = collections.defaultdict(dict)
        for k,v in out.items():
            realout[k] = {
                'points':[x['point'] for x in v],
                'indexes':[x['index'] for x in v]
            }
            
        with open(self.args.work_path+'result_global.pkl','wb') as f:
            pickle.dump(realout,f)
    
    def inspect_barcode_plotly(self, barcodes = None,html_name='result.html'):

        def retrieve_colormap(self,barcodes):
            n = len(barcodes)
            color_function = plt.cm.get_cmap('hsv', n)
            color_candidates = [color_function(i) for i in range(n)]
            color_map = {k:'rgb({},{},{})'.format(v[0]*255,v[1]*255,v[2]*255) for k,v in zip(barcodes,color_candidates)}
            return color_map


        color_map = retrieve_colormap(self,barcodes)
        coordinate = self.retrieve_coordinate()
        with open(self.args.work_path+'result_global.pkl','rb') as f:
            out = pickle.load(f)


        fig = go.Figure()

        for barcode in barcodes:

            matching = self.template_matching(barcode)
            points = np.asarray(out[barcode]['points'])
            
            fig.add_trace(go.Scatter3d(
                            z= points[:,0],
                            y= points[:,2],
                            x= points[:,1],
                            mode = 'markers',
                            # text = ['barcode:'+barcode +'<br>'+ line for line in out[barcode]['indexes']],
                            text = ['barcode:'+barcode+'<br>match:'+''.join(matching) +'<br>'+ line for line in out[barcode]['indexes']],
                            name = barcode,
                            marker = dict(
                                color = color_map[barcode],
                                size = 1,
                                opacity = 1,
                            ),
                            showlegend = True,
                            legendgroup = barcode,
                            hoverinfo= 'text',
                            # hovertemplate = '%{text} %{name}'
                        ))

        #Show 405           
        step = 20 
        for fov in self.args.fovs:  
            for zz in range(50,470,30):
                with h5py.File(self.args.h5_path.format(2,fov), "r") as f:
                    im = f[self.args.channel_names[4]][zz,0:2048:step,0:2048:step]
                    im = np.squeeze(im)

                n,_ = im.shape
                y = np.linspace(coordinate[fov,0], coordinate[fov,0]+2048,n)
                x = np.linspace(coordinate[fov,1], coordinate[fov,1]+2048,n) 
                z = np.ones((len(y),len(x))) * zz *0.4/0.1625
                fig.add_trace(go.Surface(x=x, y=y, z=z,
                    surfacecolor=im.T,
                    cmin=0, 
                    cmax=400,
                    colorscale='Blues',
                    showscale=False,
                    opacity = 0.2,
                ))
            
            fig.add_trace(go.Scatter3d(
                            z= [50],
                            y= [1024 + coordinate[fov,0]],
                            x= [1024 + coordinate[fov,1]],
                            text = str(fov),
                            mode = 'text',
                            name = 'fov',
                            showlegend = False,
                            marker = dict(
                                color = 'gray',
                                size=5,
                                opacity = 0.5,
                            )
                        ))

        # ---------------------
        fig.update_layout(
            title="My 3D scatter plot",
            width=1920,
            height=1080,
            # showlegend=False,
            scene=dict(
                aspectmode = 'data',
                xaxis_visible=True,
                yaxis_visible=True, 
                zaxis_visible=True, 
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z" ,
            ))

        # fig.show()
        fig.write_html(html_name)


    ######################################################### 
    ### ======== Unused ============
    def transform_405_single_ref_adjacent(self,code,fov,ref_code,ref_fov):
        
        '''
        self.transform_405_single_ref_adjacent(
                code = 
                ,fov = 
                ,ref_code = 
                ,ref_fov = 
                )
        '''
        
        if code == self.args.ref_code:
        
            fix_vol = nd2ToVol(self.args.fix_path, fov)
            with h5py.File(self.args.out_dir + 'code{}/{}.h5'.format(self.args.ref_code,fov), 'w') as f:
                f.create_dataset('405', fix_vol.shape, compression="gzip", dtype=fix_vol.dtype, data = fix_vol)
                
        else:

            cfg = load_cfg()
            align = alignBuild(cfg)
            align.buildSitkTile()

            with h5py.File(self.args.h5_path.format(self.args.ref_code,fov), 'r+') as f:
                fix_vol = f['405'][:]
            mov_vol = nd2ToVol(self.args.mov_path.format(code,'405',4), fov)

            try:
                initial_transform_file = self.args.out_dir + 'code{}/tforms/{}.txt'.format(ref_code,ref_fov)
                tform = align.computeTransformMap(fix_vol, mov_vol, initial_transform_file)
                align.writeTransformMap(self.args.out_dir + 'code{}/tforms/{}.txt'.format(code,fov), tform)
                
                result = align.warpVolume(mov_vol, tform)
                with h5py.File(self.args.h5_path.format(code,fov), 'w') as f:
                    f.create_dataset('405', result.shape, compression="gzip", dtype=result.dtype, data = result)
            except:
                print('failed')
    
        try:
            os.system('chmod -R 777 {}'.format(self.args.work_path))
            os.system('chmod -R 777 {}'.format(self.args.out_path))
        except:
            pass
        
    def transform_405(self,fov_code_pairs):
        
        '''
        exseq.transform_405(
            fov_code_pairs = [[30,0],[20,3]
            )
        '''
        def transform_405_single(self,code,fov):
        
            '''
            exseq.transform_405_single(
                    code = 
                    ,fov = 
                    )
            '''

            if code == self.args.ref_code:

                fix_vol = nd2ToVol(self.args.fix_path, fov)
                with h5py.File(self.args.out_dir + 'code{}/{}.h5'.format(self.args.ref_code,fov), 'w') as f:
                    f.create_dataset('405', fix_vol.shape, compression="gzip", dtype=fix_vol.dtype, data = fix_vol)

            else:

                cfg = load_cfg()
                align = alignBuild(cfg)
                align.buildSitkTile()

                with h5py.File(self.args.h5_path.format(self.args.ref_code,fov), 'r+') as f:
                    fix_vol = f['405'][:]
                mov_vol = nd2ToVol(self.args.mov_path.format(code,'405',4), fov)

                try:
                    tform = align.computeTransformMap(fix_vol, mov_vol)
                    align.writeTransformMap(self.args.out_dir + 'code{}/tforms/{}.txt'.format(code,fov), tform)

                    result = align.warpVolume(mov_vol, tform)
                    with h5py.File(self.args.h5_path.format(code,fov), 'w') as f:
                        f.create_dataset('405', result.shape, compression="gzip", dtype=result.dtype, data = result)
                except:
                    print('failed')

            try:
                os.system('chmod -R 777 {}'.format(self.args.work_path))
                os.system('chmod -R 777 {}'.format(self.args.out_path))
            except:
                pass

        for fov,code in fov_code_pairs:
            self.transform_405_single(code,fov)
            
    def transform_others_single(self,code,fov):
        
        '''transform_others_single(self,code,fov)'''

        cfg = load_cfg()
        align = alignBuild(cfg)
        align.buildSitkTile()

        ### Reference round
        if code == self.args.ref_code:
            for channel_ind, channel in enumerate(self.args.channel_names[:-1]):
                path = self.args.mov_path.format(code, channel, channel_ind)
                result = nd2ToVol(path, fov, channel)
                with h5py.File(self.args.out_dir + 'code{}/{}.h5'.format(code, fov), 'r+') as f:
                    if channel in f.keys():
                        del f[channel]
                    f.create_dataset(channel, result.shape, compression="gzip", dtype=result.dtype, data = result)
        
        ### Other rounds
        else:
            tform = align.readTransformMap(self.args.out_dir + 'code{}/tforms/{}.txt'.format(code,fov))
            for channel_ind, channel in enumerate(self.args.channel_names[:-1]):
                path = self.args.mov_path.format(code, channel,channel_ind)
                vol = nd2ToVol(path, fov, channel)
                result = align.warpVolume(vol, tform)
                with h5py.File(self.arggs.out_dir + 'code{}/{}.h5'.format(code, fov), 'r+') as f:
                    if channel in f.keys():
                        del f[channel]
                    f.create_dataset(channel, result.shape, compression="gzip", dtype=result.dtype, data = result)
                    
        try:
            os.system('chmod -R 777 {}'.format(self.args.work_path))
            os.system('chmod -R 777 {}'.format(self.args.out_path))
        except:
            pass

    def transform_others(self,fov_code_pairs):
        
        '''
        exseq.transform_others(
            fov_code_pairs = [[30,0],[20,3]
            )
        '''
        
        for fov,code in fov_code_pairs:
            self.transform_others_single(code,fov)   
    