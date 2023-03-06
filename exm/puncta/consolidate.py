from multiprocessing import current_process
import queue # imported for using queue.Empty exception
import h5py
import time
import os
from scipy.spatial.distance import cdist
import cupy as cp
import numpy as np
import multiprocessing
from multiprocessing import Process,Queue
import pickle

def consolidate_channels_function(args,fov,code):

    def find_matching_points(point_cloud1,point_cloud2,distance_threshold=8):

        temp1 = np.copy(point_cloud1)
        temp1[:,0] = temp1[:,0] * 0.5
        temp2 = np.copy(point_cloud2)
        temp2[:,0] = temp2[:,0] * 0.5

        distance = cdist(temp1, temp2, 'euclidean')
        index1 = np.argmin(distance, axis = 1)
        index2 = np.argmin(distance, axis = 0)

        valid = [i for i,x in enumerate(index1) if index2[x] == i]

        pairs = [{'point0':i,'point1':index1[i]} for i in valid 
                        if (distance[i,index1[i]] < distance_threshold)]

        return pairs

    print('Consolidate channels fov={},code={}'.format(fov,code))
    with open(args.project_path + 'processed/fov{}/coords_total_code{}.pkl'.format(fov,code), 'rb') as f:
        coords_total = pickle.load(f)

    ### 640
    reference = [{'position':position,'c0':{'index':i,'position':position}} for i, position in enumerate(coords_total['c0']) ]

    ### Other channels
    for c in [1,2,3]:

        point_cloud1 = np.asarray([x['position'] for x in reference])
        point_cloud2 = np.asarray(coords_total['c{}'.format(c)])
        pairs = find_matching_points(point_cloud1,point_cloud2)

        for pair in pairs:
            reference[pair['point0']]['c{}'.format(c)] = {
                                'index': pair['point1'],
                                'position': point_cloud2[pair['point1']]
                            }

        others = set(range(len(point_cloud2)))-set([ pair['point1'] for pair in pairs ])
        for other in others:
            reference.append({
                            'position':point_cloud2[other],
                            'c{}'.format(c) :{
                                'index': other,
                                'position': point_cloud2[other]
                            }
                        })

    with h5py.File(args.h5_path.format(code,fov), 'r') as f:
        for i, duplet in enumerate(reference):
            temp = [ f[args.channel_names[c] ][tuple(duplet['c{}'.format(c)]['position'])] if 'c{}'.format(c) in duplet else 0 for c in range(4) ]            
                        
            duplet['color'] = np.argmax(temp)
            duplet['intensity'] = temp
            duplet['index'] = i
            duplet['position'] = duplet['c{}'.format(duplet['color'])]['position']
                            
    with open(args.project_path +'processed/fov{}/result_code{}.pkl'.format(fov,code), 'wb') as f:
        pickle.dump(reference,f)
     

def consolidate_channels(self,fov_code_pairs):

    '''
    exseq.consolidate_channels(
                fov_code_pairs = [[30,0],[20,2]]
                )
    '''
                
    def f(tasks_queue,q_lock):
    
        while True: # Check for remaining task in the Queue
            try:
                with q_lock:
                    fov,code = tasks_queue.get_nowait()
                    print('Remaining tasks to process : {}'.format(tasks_queue.qsize()))
            except queue.Empty:
                print("No task left for "+ current_process().name)
                break
            else:
                self.consolidate_channels_function(fov,code)
                print('finish fov{},code{}'.format(fov,code))        
                
                
    os.environ["OMP_NUM_THREADS"] =  "1"

    # Use a quarter of the available CPU resources to finish the tasks; you can increase this if the server is accessible for this task only.
    cpu_execution_core = 20 #multiprocessing.cpu_count() / 4

    # List to hold the child processes.
    child_processes = []
    
    # Queue to hold all the puncta extraction tasks.
    tasks_queue = Queue() 
    
    # Queue lock to avoid race condition.
    q_lock = multiprocessing.Lock()
    
    # Get the extraction tasks starting time. 
    start_time = time.time()
    
    # Clear the child processes list.
    child_processes = [] 
    
    # Add all the transform_other_channels to the queue.
    for fov,code in fov_code_pairs:
        tasks_queue.put((fov,code))

    for w in range(int(cpu_execution_core)):
        p = Process(target=f, args=(tasks_queue,q_lock))
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()
        
### ================== Consolidate Codes ===================
def consolidate_codes(args,fovs,codes=range(7)):
        
    '''
    exseq.consolidate_code(
                fovs = [1,2,3]
                ,codes = [0,1,4,5]
                )
    '''
    def consolidate_codes_function(fov,codes):

        def find_matching_points(point_cloud1,point_cloud2,distance_threshold=14):

            temp1 = np.copy(point_cloud1)
            temp1[:,0] = temp1[:,0] * 0.5
            temp2 = np.copy(point_cloud2)
            temp2[:,0] = temp2[:,0] * 0.5

            distance = cdist(temp1, temp2, 'euclidean')
            index1 = np.argmin(distance, axis = 1)
            index2 = np.argmin(distance, axis = 0)

            valid = [i for i,x in enumerate(index1) if index2[x] == i]

            pairs = [{'point0':i,'point1':index1[i]} for i in valid if distance[i,index1[i]] < distance_threshold]

            return pairs

        code = 0
        with open(args.project_path + 'processed/fov{}/result_code{}.pkl'.format(fov,code), 'rb') as f:
            new = pickle.load(f)

        reference = [ { 'position': x['position'], 'code0':x } for x in new ] 

        for code in set(codes)-set([0]):

            with open(args.project_path + 'processed/fov{}/result_code{}.pkl'.format(fov,code), 'rb') as f:
                new = pickle.load(f)

            point_cloud1 = np.asarray([x['position'] for x in reference])
            point_cloud2 = np.asarray([x['position'] for x in new])

            pairs = find_matching_points(point_cloud1,point_cloud2)

            for pair in pairs:
                reference[pair['point0']]['code{}'.format(code)] = new[pair['point1']]

            others = set(range(len(point_cloud2)))-set([ pair['point1'] for pair in pairs ])
            for other in others:
                reference.append({
                    'position':point_cloud2[other],
                    'code{}'.format(code) : new[other]
                })

        reference = [ {**x,'index':i} for i,x in enumerate(reference) ]
            
        reference = [ {**x, 'barcode': ''.join([str(x['code{}'.format(code)]['color']) if 'code{}'.format(code) in x else '_' for code in args.codes ]) } for x in reference ]

        with open(args.project_path + 'processed/fov{}/result.pkl'.format(fov), 'wb') as f:
            pickle.dump(reference,f)

    for fov in fovs:
        print('Time{} Consolidate Code fov={}'.format(time.time(),fov))
        consolidate_codes_function(fov,codes)
            