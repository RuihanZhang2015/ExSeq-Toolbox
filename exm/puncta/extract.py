from multiprocessing import current_process
import queue # imported for using queue.Empty exception
import h5py
import time
import cupy as cp
import numpy as np
import multiprocessing
from multiprocessing import Process,Queue
import pickle
import collections

### ============= Extraction Per Channel =================

def calculate_coords_total_gpu(self, tasks_queue,device,lock,queue_lock,chunk_size):

    from cupyx.scipy.ndimage import gaussian_filter
    from cucim.skimage.feature import peak_local_max

    ###########
    self.args.h5_path = '/mp/nas3/ruihan/20221218_zebrafish/processed/code{}/{}_transformed.h5'
    ##########
    
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

                with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                    num_z = len(f[self.args.channel_names[0]][:,0,0])

                for c in range(4):

                    for chunk in range((num_z//chunk_size)+1):

                        with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                                img = f[self.args.channel_names[c]][max(chunk_size*chunk-7,0):min(chunk_size*(chunk+1)+7,num_z),:,:]
                                f.close()

                        with lock:
                            img = cp.array(img)
                            gaussian_filter(img, 1, output=img, mode='reflect', cval=0)
                            coords = np.array(peak_local_max(img, min_distance = 7, threshold_abs=self.args.thresholds[c],exclude_border=False).get())

                            #offset the z-axis between chunks
                            coords[:,0] += max(chunk_size*chunk-7,0)
                                    
                            # concat puncta in each chunk
                            if chunk == 0:
                                coords_total['c{}'.format(c)] = coords
                            else:     
                                coords_total['c{}'.format(c)] = np.concatenate((coords_total['c{}'.format(c)],coords),axis=0)

                            del img
                            del coords
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()


                # Remove duplicated puncta resulted from in the mautal regions between chunks. 
                for c in range(4):
                    coords_total['c{}'.format(c)] = np.unique(coords_total['c{}'.format(c)], axis=0)

                with open(self.args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'wb') as f:
                    pickle.dump(coords_total,f)
                    f.close()

            print('------ Fov:{}, Code:{} Finished on {}\n'.format(fov,code,current_process().name))


def puncta_extraction_gpu(self, tasks_queue, num_gpu, chunk_size):
            
    
    # List to hold the child processes.
    child_processes = [] 

    # Get the extraction tasks starting time. 
    start_time = time.time()

    # Queue locks to avoid race condition.
    q_lock = multiprocessing.Lock()

    print('Total tasks to process : {}'.format(tasks_queue.qsize()))

    # Excute the extraction tasks on GPU
    gpu_locks=[]
    for gpu in range(num_gpu):
        lock = multiprocessing.Lock()
        gpu_locks.append((gpu,lock))

    # Create and start a parallel execution processes based on the number of GPUs and 'process_per_gpu'. 
    process_per_gpu = 1
    for gpu_device in gpu_locks:
        for cpu_cores in range(process_per_gpu):
            p = Process(target=calculate_coords_total_gpu, args=(self, tasks_queue,int(gpu_device[0]),gpu_device[1],q_lock,chunk_size))
            child_processes.append(p)
            p.start()

    # End all the execution child processes.
    for p in child_processes:
        p.join()

    ## Show the total execution time.  
    print("total_processing_time",time.time()-start_time,"s")


def calculate_coords_total(self,tasks_queue,queue_lock,chunk_size):

    ###########
    self.args.h5_path = '/mp/nas3/ruihan/20221218_zebrafish/processed/code{}/{}_transformed.h5'
    ##########
    

    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max

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

            with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                num_z = len(f[self.args.channel_names[0]][:,0,0])

            for c in range(4):

                for chunk in range((num_z//chunk_size)+1):

                    with h5py.File(self.args.h5_path.format(code,fov), "r") as f:
                        img = f[self.args.channel_names[c]][max(chunk_size*chunk-7,0):min(chunk_size*(chunk+1)+7,num_z),:,:]
                        f.close()

                    gaussian_filter(img, 1, output=img, mode='reflect', cval=0)
                    coords = peak_local_max(img, min_distance = 7, threshold_abs=self.args.thresholds[c],exclude_border=False)

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

            with open(self.args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'wb') as f:
                print('yes')
                pickle.dump(coords_total,f)
                f.close()

        print('Fov:{}, Code:{} Finished on'.format(fov,code),current_process().name)


def puncta_extraction_cpu(self,tasks_queue,num_cpu,chunk_size):

    # List to hold the child processes.
    child_processes = [] 

    # Get the extraction tasks starting time. 
    start_time = time.time()

    # Queue locks to avoid race condition.
    q_lock = multiprocessing.Lock()

    print('Total tasks to process : {}'.format(tasks_queue.qsize()))
    # Execute the extraction tasks on the CPU only.
    # Create and start a parallel execution processes based on the number of 'cpu_execution_core'. 
    for w in range(int(num_cpu)):
        p = Process(target=calculate_coords_total, args=(self,tasks_queue,q_lock,chunk_size))
        child_processes.append(p)
        p.start()

    # End all the execution child processes.
    for p in child_processes:
        p.join()

    ## Show the total execution time.  
    print("total_processing_time",time.time()-start_time,"s")
            
        
def extract(self,fov_code_pairs,use_gpu=False,num_gpu = 3,num_cpu = 3,chunk_size=100):

    # Queue to hold all the puncta extraction tasks.
    tasks_queue = Queue() 
    
    # Add all the extraction tasks to the queue.
    for fov,code in fov_code_pairs:
        tasks_queue.put((fov,code))
        
    if use_gpu:
        processing_device = 'GPU'
        puncta_extraction_gpu(self,tasks_queue,num_gpu,chunk_size)
    else:
        processing_device = 'CPU'
        puncta_extraction_cpu(self,tasks_queue,num_cpu,chunk_size)
        
    # self.consolidate_channels(fov_code_pairs)