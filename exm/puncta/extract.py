def calculate_coords_gpu(args, tasks_queue,device,lock,queue_lock):

    import cupy as cp
    import queue # imported for using queue.Empty exception
    from cupyx.scipy.ndimage import gaussian_filter
    from multiprocessing import current_process
    from cucim.skimage.feature import peak_local_max
    import multiprocessing
    import h5py
    import pickle

    chunk_size = 100
    
    with cp.cuda.Device(device):

        while True: # Check for remaining task in the Queue

            try:
                with queue_lock:
                    temp_args = tasks_queue.get_nowait()
                    print('Remaining tasks to process : {}\n'.format(tasks_queue.qsize()))
            except  queue.Empty:
                print("No task left for {}\n".format(current_process().name))
                break
            else:
                fov,code = temp_args
                print('=======Starting Fov:{}, Code:{} on {}\n'.format(fov,code,current_process().name))   

                coords_total = dict()

                with h5py.File(args.h5_path.format(code,fov), "r") as f:
                    num_z = len(f[args.channel_names[0]][:,0,0])

                for c in range(4):

                    for chunk in range((num_z//chunk_size)+1):

                        with h5py.File(args.h5_path.format(code,fov), "r") as f:
                                img = f[args.channel_names[c]][max(chunk_size*chunk-7,0):min(chunk_size*(chunk+1)+7,num_z),:,:]
                                f.close()

                        with lock:
                            img = cp.array(img)
                            gaussian_filter(img, 1, output=img, mode='reflect', cval=0)
                            coords = cp.array(peak_local_max(img, min_distance = 7, threshold_abs=args.thresholds[c],exclude_border=False).get())

                            #offset the z-axis between chunks
                            coords[:,0] += max(chunk_size*chunk-7,0)
                                    
                            # concat puncta in each chunk
                            if chunk == 0:
                                coords_total['c{}'.format(c)] = coords
                            else:     
                                coords_total['c{}'.format(c)] = cp.concatenate((coords_total['c{}'.format(c)],coords),axis=0)

                            del img
                            del coords
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()


                # Remove duplicated puncta resulted from in the mautal regions between chunks. 
                for c in range(4):
                    coords_total['c{}'.format(c)] = cp.unique(coords_total['c{}'.format(c)], axis=0)

                with open(args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'wb') as f:
                    pickle.dump(coords_total,f)
                    f.close()
                args.chmod()
            print('------ Fov:{}, Code:{} Finished on {}\n'.format(fov,code, current_process().name))


def puncta_extraction_gpu(args, tasks_queue, num_gpu):
            
    import time
    from multiprocessing import Lock, Process
    # List to hold the child processes.
    child_processes = [] 

    # Get the extraction tasks starting time. 
    start_time = time.time()

    # Queue locks to avoid race condition.
    q_lock = Lock()

    print('Total tasks to process : {}'.format(tasks_queue.qsize()))

    # Excute the extraction tasks on GPU
    gpu_locks=[]
    for gpu in range(num_gpu):
        lock = Lock()
        gpu_locks.append((gpu,lock))

    # Create and start a parallel execution processes based on the number of GPUs and 'process_per_gpu'. 
    process_per_gpu = 1
    for gpu_device in gpu_locks:
        for cpu_cores in range(process_per_gpu):
            p = Process(target=calculate_coords_gpu, args=(args, tasks_queue,int(gpu_device[0]),gpu_device[1],q_lock))
            child_processes.append(p)
            p.start()

    # End all the execution child processes.
    for p in child_processes:
        p.join()

    ## Show the total execution time.  
    print("total_processing_time",time.time()-start_time,"s")


def calculate_coords_cpu(args,tasks_queue,queue_lock):

    import numpy as np
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    import queue # imported for using queue.Empty exception
    import collections
    from multiprocessing import current_process
    import h5py
    import pickle

    chunk_size = 100
    while True: # Check for remaining task in the Queues
        try:
            with queue_lock:
                temp_args = tasks_queue.get_nowait()
                print('Remaining tasks to process : {}'.format(tasks_queue.qsize()))
        except queue.Empty:
            print("No task left for "+ current_process().name)
            break

        else:

            fov,code = temp_args
            print('Starting Fov:{}, Code:{} on '.format(fov,code),current_process().name) 
                        
            coords_total = collections.defaultdict(list)

            with h5py.File(args.h5_path.format(code,fov), "r") as f:
                num_z = len(f[args.channel_names[0]][:,0,0])

            for c in range(4):

                for chunk in range((num_z//chunk_size)+1):

                    with h5py.File(args.h5_path.format(code,fov), "r") as f:
                        img = f[args.channel_names[c]][max(chunk_size*chunk-7,0):min(chunk_size*(chunk+1)+7,num_z),:,:]
                        f.close()

                    gaussian_filter(img, 1, output=img, mode='reflect', cval=0)
                    coords = peak_local_max(img, min_distance = 7, threshold_abs= args.thresholds[c],exclude_border=False)

                    #offset the z-axis between chunks
                    coords[:,0] += max(chunk_size*chunk-7,0)

                    # concat puncta in each chunk
                    if chunk == 0 or len(coords_total['c{}'.format(c)])==0:
                        coords_total['c{}'.format(c)] = coords
                    else:     
                        coords_total['c{}'.format(c)] = np.concatenate((coords_total['c{}'.format(c)],coords),axis=0)

            # Remove duplicated puncta resulted from the mutual regions between chunks.
            for c in range(4):
                coords_total['c{}'.format(c)] = np.unique(coords_total['c{}'.format(c)], axis=0)

            with open(args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'wb') as f:
                pickle.dump(coords_total,f)
                f.close()
            args.chmod()

        print('Fov:{}, Code:{} Finished on'.format(fov,code),current_process().name)


def puncta_extraction_cpu(args,tasks_queue,num_cpu):

    import time
    from multiprocessing import Lock, Process

    # List to hold the child processes.
    child_processes = [] 

    # Get the extraction tasks starting time. 
    start_time = time.time()

    # Queue locks to avoid race condition.
    q_lock = Lock()

    print('Total tasks to process : {}'.format(tasks_queue.qsize()))
    # Execute the extraction tasks on the CPU only.
    # Create and start a parallel execution processes based on the number of 'cpu_execution_core'. 
    for w in range(int(num_cpu)):
        p = Process(target=calculate_coords_cpu, args=(args,tasks_queue,q_lock))
        child_processes.append(p)
        p.start()

    # End all the execution child processes.
    for p in child_processes:
        p.join()

    ## Show the total execution time.  
    print("total_processing_time",time.time()-start_time,"s")


def consolidate_channels_function(args,fov,code):

    import numpy as np
    from scipy.spatial.distance import cdist
    import pickle
    import h5py

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
    with open(args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'rb') as f:
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
                            
    with open(args.work_path +'/fov{}/result_code{}.pkl'.format(fov,code), 'wb') as f:
        pickle.dump(reference,f)
    args.chmod()


def consolidate_channels(args,fov_code_pairs):

    '''
    exseq.consolidate_channels(
                fov_code_pairs = [[30,0],[20,2]]
                )
    '''

    import queue # imported for using queue.Empty exception
    from multiprocessing import current_process,Process,Queue,Lock
    import time
    import os

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
                consolidate_channels_function(args,fov,code)
                print('finish fov{},code{}'.format(fov,code))        
                
                
    os.environ["OMP_NUM_THREADS"] =  "1"

    # Use a quarter of the available CPU resources to finish the tasks; you can increase this if the server is accessible for this task only.
    cpu_execution_core = 20 #multiprocessing.cpu_count() / 4

    # List to hold the child processes.
    child_processes = []
    
    # Queue to hold all the puncta extraction tasks.
    tasks_queue = Queue() 
    
    # Queue lock to avoid race condition.
    q_lock = Lock()
    
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
        
   
def extract(args,fov_code_pairs,use_gpu=False,num_gpu = 3,num_cpu = 3):

    from multiprocessing import Queue
    import os
    # Queue to hold all the puncta extraction tasks.
    tasks_queue = Queue() 
    
    # Add all the extraction tasks to the queue.
    # for fov,code in fov_code_pairs:
    #     tasks_queue.put((fov,code))
    #     if not os.path.exists(args.work_path + 'fov{}/'.format(fov)):
    #         os.makedirs(args.work_path + 'fov{}/'.format(fov))

    # if use_gpu:
    #     processing_device = 'GPU'
    #     puncta_extraction_gpu(args,tasks_queue,num_gpu)
    # else:
    #     processing_device = 'CPU'
    #     puncta_extraction_cpu(args,tasks_queue,num_cpu)
    
    consolidate_channels(args,fov_code_pairs)


def consolidate_codes(args,fovs):
        
    '''
    exseq.consolidate_code(
                fovs = [1,2,3]
                ,codes = [0,1,4,5]
                )
    '''

    import numpy as np
    import pickle
    from scipy.spatial.distance import cdist

    def consolidate_codes_function(fov):

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
        with open(args.work_path + '/fov{}/result_code{}.pkl'.format(fov,code), 'rb') as f:
            new = pickle.load(f)

        reference = [ { 'position': x['position'], 'code0':x } for x in new ] 

        for code in range(1,7):

            print('Code = {}'.format(code))
            with open(args.work_path + '/fov{}/result_code{}.pkl'.format(fov,code), 'rb') as f:
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

        with open(args.work_path + '/fov{}/result.pkl'.format(fov), 'wb') as f:
            pickle.dump(reference,f)
        args.chmod()

    for fov in fovs:
        print('Consolidate Code fov={}'.format(fov))
        consolidate_codes_function(fov)
            
         