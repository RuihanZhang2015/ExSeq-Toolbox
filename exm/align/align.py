import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import h5py
from exm.io.io import nd2ToVol
from hijack import *
import os
from multiprocessing import current_process
import queue # imported for using queue.Empty exception
import time
import cupy as cp
import multiprocessing
from multiprocessing import Process,Queue
import pickle
import collections


def transform_405_truncated(self,code_fov_pairs):

    ref_code = self.args.ref_code
    mov_path = self.args.mov_path
    out_dir = self.args.out_dir

    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    for code,fov in code_fov_pairs:

        if tuple([code,fov]) not in starting:
            continue
        print(code,fov)
        fix_start,mov_start,last = starting[tuple([code,fov])]

        ## fix volume
        h5_name = out_dir + '/code{}/{}.h5'.format(ref_code,fov)
        if os.path.exists(h5_name):
            with h5py.File(h5_name, 'r+') as f:
                fix_vol = f['405'][fix_start:fix_start+last,:,:]
        else:
            fix_vol = nd2ToVol(mov_path.format(ref_code,'405',4), fov)
            with h5py.File(h5_name, 'w') as f:
                f.create_dataset('405', fix_vol.shape, dtype=fix_vol.dtype, data = fix_vol)
            fix_vol = fix_vol[fix_start:fix_start+last,:,:]

        ## mov volume
        h5_name = out_dir + '/code{}/{}.h5'.format(code,fov)
        if os.path.exists(h5_name):
            with h5py.File(out_dir + 'code{}/{}.h5'.format(code,fov), 'r+') as f:
                mov_vol = f['405'][mov_start:mov_start+last,:,:]
        else:
            mov_vol = nd2ToVol(mov_path.format(code,'405',4), fov)
            with h5py.File(h5_name, 'w') as f:
                f.create_dataset('405', mov_vol.shape, dtype=mov_vol.dtype, data = mov_vol)
            mov_vol = mov_vol[mov_start:mov_start+last,:,:]

        ## Align

        elastixImageFilter = sitk.ElastixImageFilter()

        fix_vol_sitk = sitk.GetImageFromArray(fix_vol)
        fix_vol_sitk.SetSpacing([1.625,1.625,4.0])
        elastixImageFilter.SetFixedImage(fix_vol_sitk)

        mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
        mov_vol_sitk.SetSpacing([1.625,1.625,4.0])
        elastixImageFilter.SetMovingImage(mov_vol_sitk)

        parameter_map = sitk.GetDefaultParameterMap('rigid')
        parameter_map['NumberOfSamplesForExactGradient'] = ['1000']  # NumberOfSamplesForExactGradient
        parameter_map['MaximumNumberOfIterations'] = ['15000'] # MaximumNumberOfIterations
        parameter_map['MaximumNumberOfSamplingAttempts'] = ['100'] # MaximumNumberOfSamplingAttempts
        parameter_map['FinalBSplineInterpolationOrder'] = ['1'] #FinalBSplineInterpolationOrder
        parameter_map['NumberOfResolutions'] = ['2']
        elastixImageFilter.SetParameterMap(parameter_map)
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.Execute()

        transform_map = elastixImageFilter.GetTransformParameterMap()
        sitk.PrintParameterMap(transform_map)
        sitk.WriteParameterFile(transform_map[0], out_dir + '/code{}/tforms/{}.txt'.format(code,fov))

        ### Apply transform

        transform_map = sitk.ReadParameterFile(out_dir + '/code{}/tforms/{}.txt'.format(code,fov))
        transformix = sitk.TransformixImageFilter()
        transformix.LogToConsoleOff()
        transformix.SetTransformParameterMap(transform_map)

        mov_vol_sitk = mov_vol_sitk[:,:,:100]

        transformix.SetMovingImage(mov_vol_sitk)
        transformix.Execute()
        out = sitk.GetArrayFromImage(transformix.GetResultImage())

        with h5py.File(out_dir + '/code{}/{}_transformed.h5'.format(code,fov), 'w') as f:
            f.create_dataset('405', out.shape, dtype=out.dtype, data = out)


# def transform_405_truncated_mask(self,code_fov_pairs):


def transform_405_full(self,code_fov_pairs):
    
    ref_code = self.args.ref_code
    mov_path = self.args.mov_path
    out_dir = self.args.out_dir
    h5_name = out_dir+'/code{}/{}.h5'.format(code,fov)

    for code,fov in code_fov_pairs:

        if tuple([code,fov]) not in starting:
            continue

        print(code,fov,'----------------------')

        fix_start,mov_start,last = starting[tuple([code,fov])]

        with h5py.File(out_dir+'/code{}/{}.h5'.format(code,fov), 'r+') as f:
            mov_vol = f['405'][:,:,:]

        mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
        mov_vol_sitk.SetSpacing([1.625,1.625,4.0])

        ### Apply transform on full res
        transform_map = sitk.ReadParameterFile(out_dir + '/code{}/tforms/{}.txt'.format(code,fov))
        print(transform_map)

        # Size
        transform_map["Size"] = tuple([str(x) for x in mov_vol.shape[::-1]])

        # Transform
        trans_um = np.array([float(x) for x in transform_map["TransformParameters"]])
        trans_um[-1] -= (fix_start-mov_start)*4
        transform_map["TransformParameters"] = tuple([str(x) for x in trans_um])     

        # center of rotation
        cen_um = np.array([float(x) for x in transform_map['CenterOfRotationPoint']])   
        cen_um[-1] += mov_start*4
        transform_map['CenterOfRotationPoint'] = tuple([str(x) for x in cen_um])  

        transformix = sitk.TransformixImageFilter()
        transformix.SetTransformParameterMap(transform_map)
        transformix.SetMovingImage(mov_vol_sitk)
        transformix.Execute()
        out = sitk.GetArrayFromImage(transformix.GetResultImage())

        with h5py.File(out_dir + '/code{}/{}_transformed.h5'.format(code,fov), 'w') as f:
            f.create_dataset('405', out.shape, dtype=out.dtype, data = out)
            
    import os
    os.system("curl -X POST -H \'Content-type: application/json\' --data \'{\"text\":\"full resolution fov7 code 1 finished!\"}\' https://hooks.slack.com/services/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")


def transform_others_function(self,tasks_queue,q_lock):

    ref_code = self.args.ref_code
    mov_path = self.args.mov_path
    out_dir = self.args.out_dir
    
    while True: # Check for remaining task in the Queue

        try:
            with q_lock:
                fov,code = tasks_queue.get_nowait()
                print('Remaining tasks to process : {}'.format(tasks_queue.qsize()))
        except queue.Empty:
            print("No task left for "+ current_process().name)
            break

        else:

            if code == self.args.ref_code:
                for channel_name_ind,channel_name in enumerate(self.args.channel_names):
                    with h5py.File(out_dir + '/code{}/{}_transformed.h5'.format(code,fov), 'a') as f:
                        if channel_name in f.keys():
                            continue
                        fix_vol = nd2ToVol(mov_path.format(code,channel_name,channel_name_ind), fov, channel_name)
                        f.create_dataset(channel_name, fix_vol.shape, dtype=fix_vol.dtype, data = fix_vol)
                continue

            if tuple([code,fov]) not in starting:
                continue
            print(code,fov,'----------------------')
            fix_start,mov_start,last = starting[tuple([code,fov])]

            for channel_name_ind,channel_name in enumerate(self.args.channel_names[:-1]):

                print(channel_name,'hello')
                with h5py.File(out_dir + '/code{}/{}_transformed.h5'.format(code,fov), 'a') as f:
                    if channel_name in f.keys():
                        continue

                mov_vol = nd2ToVol(mov_path.format(code,channel_name,channel_name_ind), fov, channel_name)

                mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
                mov_vol_sitk.SetSpacing([1.625,1.625,4.0])

                ### Apply transform on full res
                transform_map = sitk.ReadParameterFile(out_dir + '/code{}/tforms/{}.txt'.format(code,fov))
                print(transform_map)

                # Size
                transform_map["Size"] = tuple([str(x) for x in mov_vol.shape[::-1]])

                # Transform
                trans_um = np.array([float(x) for x in transform_map["TransformParameters"]])
                trans_um[-1] -= (fix_start-mov_start)*4
                transform_map["TransformParameters"] = tuple([str(x) for x in trans_um])     

                # center of rotation
                cen_um = np.array([float(x) for x in transform_map['CenterOfRotationPoint']])   
                cen_um[-1] += mov_start*4
                transform_map['CenterOfRotationPoint'] = tuple([str(x) for x in cen_um])  

                transformix = sitk.TransformixImageFilter()
                transformix.SetTransformParameterMap(transform_map)
                transformix.SetMovingImage(mov_vol_sitk)
                transformix.Execute()
                out = sitk.GetArrayFromImage(transformix.GetResultImage())

                with h5py.File(out_dir + '/code{}/{}_transformed.h5'.format(code,fov), 'a') as f:
                    f.create_dataset(channel_name, out.shape, dtype=out.dtype, data = out)
                    
    import os
    os.system("curl -X POST -H \'Content-type: application/json\' --data \'{\"text\":\"full resolution fov7 code 1 finished!\"}\' https://hooks.slack.com/services/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")



def transform_others_full(self,itfov_code_pairs,num_cpu=8):
        
    os.environ["OMP_NUM_THREADS"] = "1"

    # Use a quarter of the available CPU resources to finish the tasks; you can increase this if the server is accessible for this task only.
    if num_cpu == None:
        cpu_execution_core = multiprocessing.cpu_count() / 4
    else:
        cpu_execution_core = num_cpu
    # List to hold the child processes.
    child_processes = [] 
    # Queue to hold all the puncta extraction tasks.
    tasks_queue = Queue() 
    # Queue lock to avoid race condition.
    q_lock = multiprocessing.Lock()
    # Get the extraction tasks starting time. 
        
    start_time = time.time()
        
    # # Clear the child processes list.
    child_processes = [] 

    # Add all the align405 to the queue.
    for fov,code in fov_code_pairs:
        tasks_queue.put((fov,code))

    for w in range(int(cpu_execution_core)):
        p = Process(target=transform_others_function, args=(self,tasks_queue,q_lock))
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()

    # with open(self.args.out_dir + '/process_time.txt','a') as f:
    #     f.write(f'Align405_other_round_time,{str(time.time()-start_time)} s\n')


def inspect_alignment(self,fov_code_pairs,temp_dir):
    
    ref_code = self.args.ref_code
    mov_path = self.args.mov_path
    out_dir = self.args.out_dir

    for fov,code in fov_code_pairs:
    
        print(fov,code)

        directory = temp_dir + '/code{}/'.format(code) 
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not tuple([code,fov]) in starting:
            continue

        fix_start,mov_start,last = starting[tuple([code,fov])]
        z_stacks = np.linspace(fix_start,fix_start+last,5)

        # ---------- Full resolution -----------------
        fig,axs = plt.subplots(2,5,figsize = (20,5))
        
        for i,z in enumerate(z_stacks):
            with h5py.File(out_dir + 'code{}/{}.h5'.format(ref_code,fov), "r") as f:
                im = f['405'][int(z),:,:]
                im = np.squeeze(im)
            axs[0,i].imshow(im,vmax = 600)
            axs[0,i].set_xlabel(z)
            axs[0,i].set_ylabel('fix')

        for i,z in enumerate(z_stacks):
            with h5py.File(out_dir + 'code{}/{}_transformed.h5'.format(code,fov), "r") as f:
                im = f['405'][int(z),:,:]
                im = np.squeeze(im)
            axs[1,i].imshow(im,vmax = 600)
            axs[1,i].set_xlabel(z)
            axs[1,i].set_ylabel('transformed')
        plt.savefig(temp_dir + '/code{}/fov{}_large.jpg'.format(code,fov))
        plt.close()

        # ------------ Top left corner-------------------
        fig,axs = plt.subplots(2,5,figsize = (20,5))
        for i,z in enumerate(z_stacks):
            with h5py.File(out_dir + 'code{}/{}.h5'.format(0,fov), "r") as f:
                im = f['405'][int(z),:300,:300]
                im = np.squeeze(im)
            axs[0,i].imshow(im,vmax = 600)
            axs[0,i].set_xlabel(z)
            axs[0,i].set_ylabel('fix')

        for i,z in enumerate(z_stacks):
            with h5py.File(out_dir + 'code{}/{}_transformed.h5'.format(code,fov), "r") as f:
                im = f['405'][int(z),:300,:300]
                im = np.squeeze(im)
            axs[1,i].imshow(im,vmax = 600)
            axs[1,i].set_xlabel(z)
            axs[1,i].set_ylabel('transformed')
        plt.savefig(temp_dir + '/code{}/fov{}_topleft.jpg'.format(code,fov))
        plt.close()

        # ------------ Bottom right corner----------
        fig,axs = plt.subplots(2,5,figsize = (20,5))
        for i,z in enumerate(z_stacks):
            with h5py.File(out_dir + 'code{}/{}.h5'.format(0,fov), "r") as f:
                im = f['405'][int(z),1700:,1700:]
                im = np.squeeze(im)
            axs[0,i].imshow(im,vmax = 600)
            axs[0,i].set_xlabel(z)
            axs[0,i].set_ylabel('fix')

        for i,z in enumerate(z_stacks):
            with h5py.File(out_dir + 'code{}/{}_transformed.h5'.format(code,fov), "r") as f:
                im = f['405'][int(z),1700:,1700:]
                im = np.squeeze(im)
            axs[1,i].imshow(im,vmax = 600)
            axs[1,i].set_xlabel(z)
            axs[1,i].set_ylabel('transformed')
        plt.savefig(temp_dir + '/code{}/fov{}_bottomright.jpg'.format(code,fov))
        plt.close()