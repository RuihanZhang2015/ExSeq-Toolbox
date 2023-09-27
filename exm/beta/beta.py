import h5py
import numpy as np

from scipy.stats import rankdata

from exm.args import Args
from exm.utils.log import configure_logger
logger = configure_logger('ExSeq-Toolbox')

# TODO evaluate the benefits of this step
def quantile_normalization(args: Args, code: int, fov: int) -> None:
    r"""
    Applies quantile normalization to the volumes aligned .h5 file.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param code: The code of the volume to be normalized.
    :type code: int
    :param fov: The field of view of the volume to be normalized.
    :type fov: int
    """
    logger.warn("This function `quantile_normalization` is experimental.")
    try:
        logger.info(
            f"Starting quantile normalization for Code: {code}, FOV: {fov}")

        channels = args.channel_names[:-1]

        with h5py.File(args.h5_path.format(code, fov), "r") as f:
            volumes = [f[channel][()] for channel in channels]

        flattened_volumes = np.concatenate(
            [vol.ravel() for vol in volumes]).reshape(-1, len(channels))

        sorted_volumes = np.sort(flattened_volumes, axis=0)
        mean_volumes = np.mean(sorted_volumes, axis=1)
        rank_volumes = np.empty(flattened_volumes.shape, dtype=int)
        for i in range(flattened_volumes.shape[1]):
            rank_volumes[:, i] = rankdata(
                flattened_volumes[:, i], method='min')

        normalized_volumes = mean_volumes[rank_volumes - 1]

        # Reshape back to original shape
        reshaped_volumes = normalized_volumes.reshape(
            len(channels), *volumes[0].shape)

        # Split into separate volumes
        separate_volumes = np.split(reshaped_volumes, len(channels), axis=0)

        for vol, channel in zip(separate_volumes, channels):
            vol = np.squeeze(vol)
            channel = channel + '_norm'
            with h5py.File(args.h5_path.format(code, fov), "a") as f:
                if channel in f.keys():
                    del f[channel]
                f.create_dataset(channel, vol.shape, dtype=vol.dtype, data=vol)

        logger.info(
            f"Quantile normalization complete for Code: {code}, FOV: {fov}")

    except Exception as e:
        print(f"Error occurred while applying quantile normalization: {e}")
        raise


from exm.io.io import nd2ToVol

#TODO Fine-tune alignment parameter
# confirm a shift in the original data
def algin_channels_function(args, tasks_queue, q_lock):
    r"""
    Applies alignment between other channels and DAPI within the same round and fov.
    """
    import queue
    import multiprocessing
    from bigstream.transform import apply_transform
    from bigstream.align import affine_align, alignment_pipeline

    while True:  # Check for remaining task in the Queue
        try:
            with q_lock:
                code, fov = tasks_queue.get_nowait()
                logger.info(
                    f"Remaining tasks to process : {tasks_queue.qsize()}")
        except queue.Empty:
            logger.info(f"{multiprocessing.current_process().name}: Done")
            break
        except Exception as e:
            logger.error(f"Error fetching task from queue: {e}")
            break
        else:
            # with h5py.File(args.h5_path.format(code, fov), "r") as f:
            #     fix_volume = f['561'][()]


            def top_percent_threshold(image: np.ndarray, percent: int) -> np.ndarray:
                threshold_value = np.percentile(image, 100 - percent)  
                output_image = np.copy(image)
                output_image[output_image < threshold_value] = 0
                return output_image
            
            rigid_kwargs = { 'metric' : 'MMI',
                        'optimizer':'LBFGSB',
                        'alignment_spacing': 0.5,
                        'shrink_factors': (8, 4, 2, 1),
                        'smooth_sigmas': (0., 0., 0., 0.),
                        'optimizer_args': {
                            'gradientConvergenceTolerance': 1e-6,  # was 1e-5
                            'numberOfIterations': 1000,  # was 500
                            'maximumNumberOfCorrections': 10,  # was 5
                            'maximumNumberOfFunctionEvaluations': 5000,  # was 2000
                            'costFunctionConvergenceFactor': 1e+6,  # was 1e+7
                                            },
                    }

            spacing = [0.40, 0.1625, 0.1625]
            init_mat = np.eye(4)

            
            fix_volume = nd2ToVol(
                        args.nd2_path.format(code, '594', 1), fov, '594')

            mov_volume = nd2ToVol(
                        args.nd2_path.format(code, '640', 0), fov, '640')
            
            affine_1 = affine_align(
                    top_percent_threshold(fix_volume,0.1), top_percent_threshold(mov_volume,0.1),
                    spacing, spacing,
                    initial_condition=init_mat,
                    # rigid=True,
                **rigid_kwargs
                )
            
            np.savetxt(args.tform_path.format(
                    code, f"fov{fov}_640_affine.mat"), affine_1)


            aligned_vol_640_1 = apply_transform(
                    fix_volume, mov_volume,
                    spacing, spacing,
                    transform_list=[affine_1,],)

            
            fix_volume = nd2ToVol(
                        args.nd2_path.format(code, '561', 2), fov, '561')
            
            mov_volume = nd2ToVol(
                        args.nd2_path.format(code, '594', 1), fov, '594')

            
            affine_2 = affine_align(
                    top_percent_threshold(fix_volume,0.1), top_percent_threshold(mov_volume,0.1),
                    spacing, spacing,
                    initial_condition=init_mat,
                    # rigid=True,
                **rigid_kwargs
                )
            
            np.savetxt(args.tform_path.format(
                    code, f"fov{fov}_594_affine.mat"), affine_2)

            
            aligned_vol_640_2 = apply_transform(
                    fix_volume, aligned_vol_640_1,
                    spacing, spacing,
                    transform_list=[affine_2,],)

            del aligned_vol_640_1

            aligned_vol_594_1 = apply_transform(
                    fix_volume, mov_volume,
                    spacing, spacing,
                    transform_list=[affine_2,],)
            
            del fix_volume
            del mov_volume

            with h5py.File(args.h5_path.format(code, fov), "a") as f:
                aligned_channel_1 = '640'
                aligned_channel_2 = '594'
                
                if '488_align' in f.keys():
                    del f['488_align']
                    del f['561_align']
                    del f['640_align']
                    del f['594_align']

                
                if aligned_channel_1 in f.keys():
                    del f[aligned_channel_1]

                if aligned_channel_2 in f.keys():
                    del f[aligned_channel_2]

                f.create_dataset(
                    aligned_channel_1, aligned_vol_640_2.shape, dtype=aligned_vol_640_2.dtype, data=aligned_vol_640_2)

                f.create_dataset(
                    aligned_channel_2, aligned_vol_594_1.shape, dtype=aligned_vol_594_1.dtype, data=aligned_vol_594_1)

                for i,channel in enumerate(['561','488','405']):
                    if channel in f.keys():
                        del f[channel]
                    volume = nd2ToVol(
                        args.nd2_path.format(code, channel, i+2), fov, channel)
                    f.create_dataset(
                    channel, volume.shape, dtype=volume.dtype, data=volume)
                    



def algin_channels(args: Args,
                   code_fov_pairs,
                   parallel_processes: int = 1) -> None:
    r"""
    Applies alignment between other channels and DAPI within the same round and fov.
    """
    logger.warn("This function `algin_channels` is experimental.")
    import multiprocessing

    child_processes = []
    tasks_queue = multiprocessing.Queue()
    q_lock = multiprocessing.Lock()

    if not code_fov_pairs:
        code_fov_pairs = [[round_val, roi_val]
                          for round_val in args.codes for roi_val in args.fovs]

    for round, roi in code_fov_pairs:
        tasks_queue.put((round, roi))

    for w in range(int(parallel_processes)):

        p = multiprocessing.Process(
            target=algin_channels_function, args=(args, tasks_queue, q_lock))

        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()




#TODO fine-tune foreground_segmentation parameter   
# test possiblity to train a u-net for each speices 
def foreground_segmentation(args):
    r"""
    Applies forground_segmentation on code 0 DAPI for each fov.
    """
    import numpy as np
    from exm.io.io import nd2ToVol
    from scipy import ndimage
    from bigstream import level_set
    import tifffile
    import os
    logger.warn("This function `forground_segmentation` is experimental.")
    def histogram_stretch(image, min_out=0, max_out=255):
        
        # Calculate the minimum and maximum pixel values in the image
        min_val = np.min(image)
        max_val = np.max(image)

        # Perform histogram stretching
        adjusted_image = ((image - min_val) / (max_val - min_val)
                        * (max_out - min_out) + min_out).astype(np.int8)

        return adjusted_image

    for fov in args.fovs:
        vol = nd2ToVol(args.nd2_path.format(0,'405', 4), fov)
        vol = histogram_stretch(vol, min_out=0, max_out=255)
        vol_cont = ndimage.zoom(vol, (1, 0.1, 0.1), order=1)
        noise = level_set.estimate_background(
            vol_cont.astype(np.uint8), int(vol_cont.shape[1]/20))
        mask = level_set.foreground_segmentation(
            vol_cont.astype(np.uint8),
            voxel_spacing = np.array([1,1,1]),
            mask_smoothing=2,
            iterations=[80, 40, 10],
            smooth_sigmas=[12, 6, 3],
            lambda2=1.0,
            background=noise,
        )
        vol_cont = ndimage.zoom(mask, (1, 10, 10), order=0)

        tifffile.imwrite(os.path.join(args.puncta_path,'fg_masks',f'fov{fov}.tif'), mask)


#TODO fine-tune foreground_segmentation parameter   
# test possiblity to train a u-net for each speices 
def foreground_segmentation_new(args):
    r"""
    Applies forground_segmentation on code 0 DAPI for each fov.
    """
    import numpy as np
    from exm.io.io import nd2ToVol
    from scipy import ndimage
    from bigstream import level_set
    import tifffile
    import os
    from skimage import morphology
    
    logger.warn("This function `foreground_segmentation_new` is experimental.")

    for fov in args.fovs:
        volume = nd2ToVol(args.nd2_path.format(0,'405', 4), fov)

        vol = ndimage.zoom(volume,(0.1,0.1,0.1),order=1)
        noise = level_set.estimate_background(vol.astype(np.uint8),int(vol.shape[1]/20))
        skip_spacing = np.array([1,1,1])
        fg_mask = level_set.foreground_segmentation(
            vol.astype(np.uint8), skip_spacing,
            shrink_factors = [4,2,1],
            mask_smoothing=2,
            iterations=[80,40,10],
            smooth_sigmas=[12,6,3],
            lambda2=1.0,
            background = noise,
            )
            
        for _ in range(10):
            fg_mask = morphology.dilation(fg_mask)
            
        upsample_factor = np.array(volume.shape) / np.array(fg_mask.shape)
        fg_mask = ndimage.zoom(fg_mask,upsample_factor,order=1) 

        tifffile.imwrite(os.path.join(args.puncta_path,'fg_masks',f'fov{fov}.tif'), fg_mask)


import os
from bigstream.align import affine_align, alignment_pipeline

#TODO Fine-tune alignment parameter
# confirm a shift in the original data
def algin_channels_function_new(args, tasks_queue, q_lock):
    r"""
    Applies alignment between other channels and DAPI within the same round and fov.
    """
    import queue
    import multiprocessing
    from bigstream.transform import apply_transform
    from bigstream.align import affine_align, alignment_pipeline

    while True:  # Check for remaining task in the Queue
        try:
            with q_lock:
                code, fov = tasks_queue.get_nowait()
                logger.info(
                    f"Remaining tasks to process : {tasks_queue.qsize()}")
        except queue.Empty:
            logger.info(f"{multiprocessing.current_process().name}: Done")
            break
        except Exception as e:
            logger.error(f"Error fetching task from queue: {e}")
            break
        else:

            def top_percent_threshold(image: np.ndarray, percent: int) -> np.ndarray:
                threshold_value = np.percentile(image, 100 - percent)  
                output_image = np.copy(image)
                output_image[output_image < threshold_value] = 0
                return output_image
            
            spacing = np.array([0.40, 0.1625, 0.1625])
            init_mat = np.eye(4)
            
            # h5_path = os.path.join(args.processed_data_path, "code{}/{}.h5")
            
            with h5py.File(args.h5_path.format(code,fov),'r') as f:
                fix_volume = f['594'][()]

            with h5py.File(args.h5_path.format(code,fov),'r') as f:
                mov_volume = f['640'][()]


            rigid_kwargs = { 'metric' : 'D',
                        'optimizer':'LBFGSB',
                        'alignment_spacing': 0.5,
                        'shrink_factors': (4, 2, 1),
                        'smooth_sigmas': (1.,1.,1.),
                        'initial_condition':'CENTER',
                        'optimizer_args': {
                            'gradientConvergenceTolerance': 1e-6,  # was 1e-5
                            'numberOfIterations': 1000,  # was 500
                            'maximumNumberOfCorrections': 10,  # was 5
                            'maximumNumberOfFunctionEvaluations': 5000,  # was 2000
                            'costFunctionConvergenceFactor': 1e+6,  # was 1e+7
                                            },
                    }


            affine_1 = affine_align(
                    fix_volume, mov_volume,
                    spacing, spacing,
                    rigid=True,
                **rigid_kwargs
                )
            if np.array_equal(affine_1, init_mat):
                logger.error(f" Align 640 -> 594 code{code} fov{fov} failed")

            # np.savetxt(args.tform_path.format(
            #         code, f"fov{fov}_640_affine.mat"), affine_1)


            aligned_vol_640_1 = apply_transform(
                    fix_volume, mov_volume,
                    spacing, spacing,
                    transform_list=[affine_1,],)

        
            with h5py.File(args.h5_path.format(code,fov),'r') as f:
                fix_volume = f['561'][()]
            

            with h5py.File(args.h5_path.format(code,fov),'r') as f:
                mov_volume = f['594'][()]

            
            affine_2 = affine_align(
                    fix_volume, mov_volume,
                    spacing, spacing,
                    rigid=True,
                **rigid_kwargs
                )
            if np.array_equal(affine_2, init_mat):
                logger.error(f" Align 594 -> 561 code{code} fov{fov} failed")

            # np.savetxt(args.tform_path.format(
            #         code, f"fov{fov}_594_affine.mat"), affine_2)

            
            aligned_vol_640_2 = apply_transform(
                    fix_volume, aligned_vol_640_1,
                    spacing, spacing,
                    transform_list=[affine_2,],)

            del aligned_vol_640_1

            aligned_vol_594_1 = apply_transform(
                    fix_volume, mov_volume,
                    spacing, spacing,
                    transform_list=[affine_2,],)
            
            del fix_volume
            del mov_volume

            with h5py.File(args.h5_path.format(code, fov), "a") as f:
            

                aligned_channel_1 = '640'
                aligned_channel_2 = '594'
                             
                if aligned_channel_1 in f.keys():
                    del f[aligned_channel_1]

                if aligned_channel_2 in f.keys():
                    del f[aligned_channel_2]

                f.create_dataset(
                    aligned_channel_1, aligned_vol_640_2.shape, dtype=aligned_vol_640_2.dtype, data=aligned_vol_640_2)

                f.create_dataset(
                    aligned_channel_2, aligned_vol_594_1.shape, dtype=aligned_vol_594_1.dtype, data=aligned_vol_594_1)

            # with h5py.File(args.h5_path.format(code,fov),'r') as f: 

            #     for i,channel in enumerate(['561','488','405']):
            #         volume = f[channel][()]
                    
            #         with h5py.File(args.h5_path.format(code, fov), "a") as x:

            #             x.create_dataset(
            #             channel, volume.shape, dtype=volume.dtype, data=volume)
                        


def algin_channels_new(args: Args,
                   code_fov_pairs,
                   parallel_processes: int = 1) -> None:
    r"""
    Applies alignment between other channels and DAPI within the same round and fov.
    """
    logger.warn("This function `algin_channels_new` is experimental.")
    import multiprocessing

    child_processes = []
    tasks_queue = multiprocessing.Queue()
    q_lock = multiprocessing.Lock()

    if not code_fov_pairs:
        code_fov_pairs = [[round_val, roi_val]
                          for round_val in args.codes for roi_val in args.fovs]

    for round, roi in code_fov_pairs:
        tasks_queue.put((round, roi))

    for w in range(int(parallel_processes)):

        p = multiprocessing.Process(
            target=algin_channels_function_new, args=(args, tasks_queue, q_lock))

        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()

