"""
Volumetric alignment module is designed to facilitate the precise alignment of volumetric microscopy data, particularly for large volumes, which are characterized by having large Z-slices. The central function, **`volumetric_alignment`**, serves as the primary interface for users to perform alignment operations. It abstracts the complexities of the underlying alignment mechanisms and offers a simple, unified entry point for processing.
"""

import h5py
import queue
import tempfile
import traceback
import multiprocessing
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

from exm.args import Args
from exm.io.io import nd2ToVol , get_raw_volume
from exm.utils import chmod, subtract_background_rolling_ball, subtract_background_top_hat,downsample_volume, enhance_and_filter_volume

from exm.utils import configure_logger
logger = configure_logger('ExSeq-Toolbox')



def transform_ref_code(args: Args, fov: int, bg_sub: str, dataset_type) -> None:
    """
    Transforms reference round for each volume specified in code_fov_pairs, convert from an nd2 file to an array, 
    then save into an .h5 file.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param fov: Field of View index.
    :type fov: int
    :param bg_sub: Background subtraction method, can be either "rolling_ball", "top_hat" or None. Default is None.
    :type bg_sub: str, optional
    """
    logger.info(f"Transform ref round: Round:{args.ref_code}, ROI:{fov}")

    for channel_ind, channel in enumerate(args.channel_names):
        try:
            with h5py.File(args.data_path.format(args.ref_code,fov), "r") as f:
                ref_vol = f[channel][()]           
            if channel == args.ref_channel and bg_sub == 'rolling_ball':
                ref_vol = subtract_background_rolling_ball(ref_vol)

            if channel == args.ref_channel and bg_sub == 'top_hat':
                ref_vol = subtract_background_top_hat(ref_vol)

            with h5py.File(args.h5_path.format(args.ref_code, fov), "a") as f:
                if channel in f.keys():
                    del f[channel]
                f.create_dataset(channel, ref_vol.shape,
                                 dtype=ref_vol.dtype, data=ref_vol)
        except Exception as e:
            logger.error(
                f"Error during transformation for  Ref Round, ROI: {fov}, Channel: {channel}, Error: {e}")
            raise


def execute_volumetric_alignment(args: Args,
                                 tasks_queue: multiprocessing.Queue,
                                 q_lock: multiprocessing.Lock,
                                 acclerated,
                                 dataset_type) -> None:
    r"""
    For each volume in code_fov_pairs, finds the corresponding reference volume and performs alignment.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param tasks_queue: A multiprocessing queue containing tasks. Each task is a tuple of (code, fov)
    :type tasks_queue: multiprocessing.Queue
    :param q_lock: A lock for synchronizing tasks queue access.
    :type q_lock: multiprocessing.Lock
    """

    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalWarningDisplay(False)

    while True:  # Check for remaining task in the Queue

        try:
            with q_lock:
                code, fov, bg_sub = tasks_queue.get_nowait()
                logger.info(
                    f"Remaining tasks to process : {tasks_queue.qsize()}")
        except queue.Empty:
            logger.info(f"{multiprocessing.current_process().name}: Done")
            break
        except Exception as e:
            logger.error(f"Error fetching task from queue: {e}")
            break
        else:
            try:
                if code == args.ref_code:
                    transform_ref_code(args, fov, bg_sub,dataset_type)

                logger.info(f"Aligning: Code:{code},Fov:{fov}")

                fix_vol = nd2ToVol(args.nd2_path.format(args.ref_code, args.ref_channel,
                                   args.channel_names.index(args.ref_channel)), fov, args.ref_channel,dataset_type=dataset_type)

                mov_vol = nd2ToVol(args.nd2_path.format(
                    code, args.ref_channel, args.channel_names.index(args.ref_channel)), fov, args.ref_channel,dataset_type=dataset_type)

                fix_vol_sitk = sitk.GetImageFromArray(fix_vol)
                fix_vol_sitk.SetSpacing(args.spacing)

                mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
                mov_vol_sitk.SetSpacing(args.spacing)

                # Initialize transform using Center of Gravity
                initial_transform = sitk.CenteredTransformInitializer(
                    fix_vol_sitk, mov_vol_sitk,
                    sitk.Euler3DTransform(),
                    sitk.CenteredTransformInitializerFilter.GEOMETRY)

                # Apply the transform to the moving image
                mov_vol_sitk = sitk.Resample(
                    mov_vol_sitk, fix_vol_sitk, initial_transform, sitk.sitkLinear, 0.0, mov_vol_sitk.GetPixelID())

                # temp dicectory for the log files
                tmpdir_obj = tempfile.TemporaryDirectory()

                # Align
                elastixImageFilter = sitk.ElastixImageFilter()
                elastixImageFilter.SetLogToFile(False)
                elastixImageFilter.SetLogToConsole(True)
                elastixImageFilter.SetOutputDirectory(tmpdir_obj.name)

                elastixImageFilter.SetFixedImage(fix_vol_sitk)
                elastixImageFilter.SetMovingImage(mov_vol_sitk)

                # Translation across x, y, and z only
                parameter_map = sitk.GetDefaultParameterMap("translation")
                parameter_map["NumberOfSamplesForExactGradient"] = ["1000"]
                parameter_map["MaximumNumberOfIterations"] = ["25000"]
                parameter_map["MaximumNumberOfSamplingAttempts"] = ["2000"]
                parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
                parameter_map["FixedImagePyramid"] = [
                    "FixedRecursiveImagePyramid"]
                parameter_map["MovingImagePyramid"] = [
                    "MovingRecursiveImagePyramid"]
                parameter_map["NumberOfResolutions"] = ["5"]
                parameter_map["FixedImagePyramidSchedule"] = [
                    "10 10 10 8 8 8 4 4 4 2 2 2 1 1 1"]
                elastixImageFilter.SetParameterMap(parameter_map)

                # Translation + rotation
                parameter_map = sitk.GetDefaultParameterMap("rigid")
                parameter_map["NumberOfSamplesForExactGradient"] = ["1000"]
                parameter_map["MaximumNumberOfIterations"] = ["25000"]
                parameter_map["MaximumNumberOfSamplingAttempts"] = ["2000"]
                parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
                parameter_map["FixedImagePyramid"] = [
                    "FixedShrinkingImagePyramid"]
                parameter_map["MovingImagePyramid"] = [
                    "MovingShrinkingImagePyramid"]
                parameter_map["NumberOfResolutions"] = ["1"]
                parameter_map["FixedImagePyramidSchedule"] = ["1 1 1"]
                parameter_map["MovingImagePyramidSchedule"] = ["1 1 1"]
                elastixImageFilter.AddParameterMap(parameter_map)

                # Translation, rotation, scaling and shearing
                parameter_map = sitk.GetDefaultParameterMap("affine")
                parameter_map["NumberOfSamplesForExactGradient"] = ["1000"]
                parameter_map["MaximumNumberOfIterations"] = ["25000"]
                parameter_map["MaximumNumberOfSamplingAttempts"] = ["2000"]

                parameter_map["FinalBSplineInterpolationOrder"] = ["1"]
                parameter_map["FixedImagePyramid"] = [
                    "FixedShrinkingImagePyramid"]
                parameter_map["MovingImagePyramid"] = [
                    "MovingShrinkingImagePyramid"]
                parameter_map["NumberOfResolutions"] = ["1"]
                parameter_map["FixedImagePyramidSchedule"] = ["1 1 1"]
                parameter_map["MovingImagePyramidSchedule"] = ["1 1 1"]
                elastixImageFilter.AddParameterMap(parameter_map)

                elastixImageFilter.Execute()

                transform_map = elastixImageFilter.GetTransformParameterMap()

                for channel_ind, channel in enumerate(args.channel_names):

                    mov_vol = nd2ToVol(args.nd2_path.format(
                        code, channel, channel_ind), fov, channel,dataset_type=dataset_type)
                    mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
                    mov_vol_sitk.SetSpacing(args.spacing)

                    transformixImageFilter = sitk.TransformixImageFilter()
                    transformixImageFilter.SetMovingImage(mov_vol_sitk)
                    transformixImageFilter.SetTransformParameterMap(
                        elastixImageFilter.GetTransformParameterMap())
                    transformixImageFilter.LogToConsoleOff()
                    transformixImageFilter.Execute()

                    out = sitk.GetArrayFromImage(
                        transformixImageFilter.GetResultImage())

                    with h5py.File(args.h5_path.format(code, fov), "a") as f:
                        if channel in f.keys():
                            del f[channel]
                        f.create_dataset(channel, out.shape,
                                         dtype=out.dtype, data=out)

                tmpdir_obj.cleanup()
            except Exception as e:
                logger.error(
                    f"Error during alignment for Code: {code}, FOV: {fov}, Error: {e}")
                raise


def execute_volumetric_alignment_bigstream(args: Args,
                                           tasks_queue: multiprocessing.Queue,
                                           q_lock: multiprocessing.Lock,
                                           acclerated,
                                           dataset_type) -> None:
    r"""
    Executes volumetric alignment using BigStream for each code and FOV from the tasks queue.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param tasks_queue: A multiprocessing queue containing tasks. Each task is a tuple of (round, roi, bg_sub).
    :type tasks_queue: multiprocessing.Queue
    :param q_lock: A lock for synchronizing tasks queue access.
    :type q_lock: multiprocessing.Lock
    :param acclerated: Flag to use gpu for image filter.
    :type q_lock: bool
    """
    
    from bigstream.transform import apply_transform 
    from bigstream.align import affine_align
    from scipy.ndimage import zoom
    from bigstream.align import alignment_pipeline

    while True:  # Check for remaining task in the Queue

        try:
            with q_lock:
                code, fov, bg_sub = tasks_queue.get_nowait()
                logger.info(
                    f"Remaining tasks to process : {tasks_queue.qsize()}")
        except queue.Empty:
            logger.info(f"{multiprocessing.current_process().name}: Done")
            break
        except Exception as e:
            logger.error(f"Error fetching task from queue: {e}")
            break
        else:
            try:
                if code == args.ref_code:
                    transform_ref_code(args, fov, bg_sub,dataset_type=dataset_type)
                    continue

                logger.info(f"aligning: Code:{code},FOV:{fov}")

                try:
                    with h5py.File(args.h5_path.format(args.ref_code, fov), "r") as f:
                        fix_vol = f[args.ref_channel][()]

                except Exception as e:
                    logger.error(
                        f"The refrence code for FOV:{fov} is not processed yet, {e}")
                    continue
                
                with h5py.File(args.data_path.format(code,fov), "r") as f:
                    mov_vol = f[args.ref_channel][()]

                if bg_sub == "rolling_ball":
                    mov_vol = subtract_background_rolling_ball(mov_vol)

                if bg_sub == "top_hat":
                    mov_vol = subtract_background_top_hat(mov_vol)

                
                downsample_factors = (2, 4, 4)
                downsample_spacing = np.array(args.spacing) * np.array(downsample_factors)

                filtered_fix_vol = enhance_and_filter_volume(fix_vol,40.,99.8, acclerated)
                filtered_mov_vol = enhance_and_filter_volume(mov_vol,40.,99.8, acclerated)

                fix_downsampled = downsample_volume(filtered_fix_vol, downsample_factors)
                move_downsampled = downsample_volume(filtered_mov_vol, downsample_factors) 
                    
                
                ransac_kwargs = {'blob_sizes': [2,200]}
                affine_kwargs = { 'metric' : 'MMI',
                                    'optimizer':'LBFGSB',
                                    'alignment_spacing': 1,
                                    'shrink_factors': ( 4, 2, 1),
                                    'smooth_sigmas': ( 0., 0., 0.),
                                     'optimizer_args':{
                                    'gradientConvergenceTolerance' :1e-6,
                                    'numberOfIterations':800,
                                    'maximumNumberOfCorrections' : 8,
                                    },
                                }

       
                affine_ransac = alignment_pipeline(fix_downsampled, move_downsampled, downsample_spacing, downsample_spacing, steps = [('ransac', ransac_kwargs)])
                affine = alignment_pipeline(filtered_fix_vol, filtered_mov_vol, np.array(args.spacing), np.array(args.spacing), steps=[('affine', affine_kwargs)],static_transform_list=[affine_ransac])

                for channel_ind, channel in enumerate(args.channel_names):

                    with h5py.File(args.data_path.format(code,fov), "r") as f:
                        mov_vol = f[channel][()]
                    aligned_vol = apply_transform(
                        fix_vol, mov_vol,
                        args.spacing, args.spacing,
                        transform_list=[affine_ransac,affine],
                    )

                    with h5py.File(args.h5_path.format(code, fov), "a") as f:
                        if channel in f.keys():
                            del f[channel]
                        f.create_dataset(channel, aligned_vol.shape,
                                         dtype=aligned_vol.dtype, data=aligned_vol)

            except Exception as e:
                logger.error(
                    f"Error during alignment for Code: {code}, ROI: {fov}, Error: {e} , {traceback.format_exc()}")
                raise


def volumetric_alignment(args: Args,
                         code_fov_pairs: Optional[List[Tuple[int, int]]] = None,
                         parallel_processes: int = 1,
                         method: Optional[str] = None,
                         bg_sub: Optional[str] = '',
                         acclerated : Optional[bool] = False,
                         dataset_type='.nd2') -> None:
    r"""
    Coordinates the alignment of volumetric data across different fields of view (FOV) and codes.
    This function sets up parallel processing to handle multiple FOV and code combinations, using either
    SITK alignment methods or BigStream, depending on the `method` parameter. Background subtraction can
    also be applied as specified.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param code_fov_pairs: A list of tuples, where each tuple is a (code, fov) pair. If None, uses all code and fov pairs from the args.
    :type code_fov_pairs: List[Tuple[int, int]], optional
    :param parallel_processes: The number of processes to use for parallel processing. Default is 1, which means no parallel processing.
    :type parallel_processes: int, optional
    :param method: The method to use for alignment. If 'bigstream', uses the 'execute_volumetric_alignment_bigstream' function. Otherwise, uses the 'execute_volumetric_alignment' function.
    :type method: str, optional
    :param bg_sub: Specifies the background subtraction method to be used. Can be "rolling_ball" or "top_hat". If not provided, no background subtraction will be applied.
    :type bg_sub: str, optional
    """

    child_processes = []
    tasks_queue = multiprocessing.Queue()
    q_lock = multiprocessing.Lock()

    if not code_fov_pairs:
        code_fov_pairs = [[code_val, fov_val]
                          for code_val in args.codes for fov_val in args.fovs]

    for code, fov in code_fov_pairs:
        tasks_queue.put((code, fov, bg_sub))

    for w in range(int(parallel_processes)):
        try:

            if method == 'bigstream':
                p = multiprocessing.Process(
                    target=execute_volumetric_alignment_bigstream, args=(args, tasks_queue, q_lock,acclerated,dataset_type))
            else:
                p = multiprocessing.Process(
                    target=execute_volumetric_alignment, args=(args, tasks_queue, q_lock,acclerated,dataset_type))

            child_processes.append(p)
            p.start()
        except Exception as e:
            logger.error(
                f"Error starting process for Code: {round}, FOV: {roi}, Error: {e}")

    for p in child_processes:
        p.join()


