import os
import h5py
import pickle
import multiprocessing
import numpy as np
import queue
from multiprocessing import current_process,Lock,Process,Queue

from exm.utils import chmod

# TODO document function
def calculate_coords_gpu(args,tasks_queue,device,lock,queue_lock):

    import cupy as cp
    import queue # imported for using queue.Empty exception
    from cupyx.scipy.ndimage import gaussian_filter
    from cucim.skimage.feature import peak_local_max

    # TODO varaible chunk size 
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
                print('calculate_coords_gpu: code{}, fov{} on {}\n'.format(fov,code,current_process().name))   

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
                    coords_total['c{}'.format(c)] = np.unique(coords_total['c{}'.format(c)], axis=0)

                with open(args.work_path + '/fov{}/coords_total_code{}.pkl'.format(fov,code), 'wb') as f:
                    pickle.dump(coords_total,f)
                    f.close()
  
                if args.permission:
                    chmod(os.path.join(args.work_path,'fov{}/coords_total_code{}.pkl'.format(fov,code)))
            print('------ Fov:{}, Code:{} Finished on {}\n'.format(fov,code, current_process().name))


def puncta_extraction_gpu(args, tasks_queue, num_gpu):
            
    # List to hold the child processes.
    child_processes = [] 
    # Queue locks to avoid race condition.
    q_lock = Lock()

    print('Total tasks to process : {}\n'.format(tasks_queue.qsize()))
    # Excute the extraction tasks on GPU
    gpu_locks=[]
    for gpu in range(num_gpu):
        lock = Lock()
        gpu_locks.append((gpu,lock))

    #Give user option to set process_per_gpu
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


def calculate_coords_cpu(args,tasks_queue,queue_lock):

    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max
    import collections
    
    chunk_size = 100
    while True: # Check for remaining task in the Queues
        try:
            with queue_lock:
                temp_args = tasks_queue.get_nowait()
                print('Remaining tasks to process : {}\n'.format(tasks_queue.qsize()))
        except queue.Empty:
            print("No task left for {}\n".format(current_process().name))
            break

        else:

            fov,code = temp_args
            print('calculate_coords_cpu: code{}, fov{} on {}\n'.format(fov,code,current_process().name)) 

            #TODO why we using collections for dict here            
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

            if args.permission:
                chmod(os.path.join(args.work_path,'fov{}/coords_total_code{}.pkl'.format(fov,code)))

        print('Extract Puncta: Fov{}, Code{} Finished on {}\n'.format(fov,code,current_process().name))


def puncta_extraction_cpu(args,tasks_queue,num_cpu):

    # List to hold the child processes.
    child_processes = [] 

    # Queue locks to avoid race condition.
    q_lock = Lock()

    print('Total tasks to process : {}\n'.format(tasks_queue.qsize()))
    # Execute the extraction tasks on the CPU only.
    # Create and start a parallel execution processes based on the number of 'num_cpu'. 
    for w in range(int(num_cpu)):
        p = Process(target=calculate_coords_cpu, args=(args,tasks_queue,q_lock))
        child_processes.append(p)
        p.start()

    # End all the execution child processes.
    for p in child_processes:
        p.join()

def extract(args,fov_code_pairs,use_gpu=False,num_gpu = 3,num_cpu = 3):

    # Queue to hold all the puncta extraction tasks.
    tasks_queue = Queue() 
    
    # Add all the extraction tasks to the queue.
    for code,fov in fov_code_pairs:
        tasks_queue.put((fov,code))
        if not os.path.exists(args.work_path + 'fov{}/'.format(fov)):
            os.makedirs(args.work_path + 'fov{}/'.format(fov))

    if use_gpu:
        processing_device = 'GPU'
        puncta_extraction_gpu(args,tasks_queue,num_gpu)
    else:
        processing_device = 'CPU'
        puncta_extraction_cpu(args,tasks_queue,num_cpu)
         