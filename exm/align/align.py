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
from exm.io.io import nd2ToVol
from exm.utils import chmod, subtract_background_rolling_ball, subtract_background_top_hat,downsample_volume, enhance_and_filter_volume

from exm.utils import configure_logger
logger = configure_logger('ExSeq-Toolbox')



def transform_ref_code(args: Args, fov: int, bg_sub: str) -> None:
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



def execute_volumetric_alignment_bigstream(args: Args,
                                           tasks_queue: multiprocessing.Queue,
                                           q_lock: multiprocessing.Lock,
                                           accelerated,
                                           **kwargs) -> None:
    r"""
    Executes volumetric alignment using BigStream for each code and FOV from the tasks queue.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param tasks_queue: A multiprocessing queue containing tasks. Each task is a tuple of (round, roi, bg_sub).
    :type tasks_queue: multiprocessing.Queue
    :param q_lock: A lock for synchronizing tasks queue access.
    :type q_lock: multiprocessing.Lock
    :param accelerated: Flag to use GPU for image filter.
    :type accelerated: bool
    :param kwargs: Additional optional parameters, including `downsample_factors`, `downsample_steps`, `full_size_steps`, and percentile values (`low`, `high`).
    """

    from bigstream.transform import apply_transform
    from bigstream.align import alignment_pipeline

    # Default parameters
    downsample_factors = kwargs.get('downsample_factors', (1, 1, 1))
    downsample_steps = kwargs.get('downsample_steps', [])
    full_size_steps = kwargs.get('full_size_steps', [])
    run_downsample_steps = kwargs.get('run_downsample_steps', True)  # Control execution of downsampling steps
    low = kwargs.get('low', 1.0)
    high = kwargs.get('high', 99.0)

    if not downsample_steps and not full_size_steps:
        raise ValueError("No alignment steps provided. Please pass `downsample_steps` or `full_size_steps`.")

    if not isinstance(downsample_factors, tuple) or len(downsample_factors) != 3:
        raise ValueError("downsample_factors must be a tuple of three integers.")
    if not isinstance(downsample_steps, list) or not isinstance(full_size_steps, list):
        raise ValueError("downsample_steps and full_size_steps must be lists of steps.")

    while True:  # Check for remaining tasks in the queue
        try:
            with q_lock:
                code, fov, bg_sub = tasks_queue.get_nowait()
                logger.info(f"Remaining tasks to process: {tasks_queue.qsize()}")
        except queue.Empty:
            logger.info(f"{multiprocessing.current_process().name}: Done")
            break
        except Exception as e:
            logger.error(f"Error fetching task from queue: {e}")
            break
        
        try:
            if code == args.ref_code:
                transform_ref_code(args, fov, bg_sub)
                continue

            logger.info(f"Aligning: Code:{code}, FOV:{fov}")

            try:
                with h5py.File(args.h5_path.format(args.ref_code, fov), "r") as f:
                    fix_vol = f[args.ref_channel][()]

            except Exception as e:
                logger.error(f"The reference code for FOV:{fov} is not processed yet: {e}")
                continue

            with h5py.File(args.data_path.format(code, fov), "r") as f:
                mov_vol = f[args.ref_channel][()]

            if bg_sub == "rolling_ball":
                mov_vol = subtract_background_rolling_ball(mov_vol)

            if bg_sub == "top_hat":
                mov_vol = subtract_background_top_hat(mov_vol)

            downsample_spacing = np.array(args.spacing) * np.array(downsample_factors)

            # Filter volumes
            filtered_fix_vol = enhance_and_filter_volume(fix_vol, low, high, accelerated)
            filtered_mov_vol = enhance_and_filter_volume(mov_vol, low, high, accelerated)

            # Initialize transformation lists
            downsample_transform = None
            full_transform = None

            # Run downsample steps if applicable
            if run_downsample_steps and downsample_factors != (1, 1, 1) and downsample_steps:
                fix_downsampled = downsample_volume(filtered_fix_vol, downsample_factors)
                move_downsampled = downsample_volume(filtered_mov_vol, downsample_factors)

                downsample_transform = alignment_pipeline(
                    fix_downsampled, move_downsampled,
                    downsample_spacing, downsample_spacing,
                    steps=downsample_steps
                )

            # Run full-size steps
            if full_size_steps:
                static_transforms = [downsample_transform] if downsample_transform is not None else []
                full_transform = alignment_pipeline(
                    filtered_fix_vol, filtered_mov_vol,
                    np.array(args.spacing), np.array(args.spacing),
                    steps=full_size_steps,
                    static_transform_list=static_transforms
                )

            # Apply transformations to all channels
            for channel in args.channel_names:
                with h5py.File(args.data_path.format(code, fov), "r") as f:
                    mov_vol = f[channel][()]
                
                transform_list = [t for t in [downsample_transform, full_transform] if t is not None]
                aligned_vol = apply_transform(
                    fix_vol, mov_vol,
                    args.spacing, args.spacing,
                    transform_list=transform_list,
                )

                with h5py.File(args.h5_path.format(code, fov), "a") as f:
                    if channel in f.keys():
                        del f[channel]
                    f.create_dataset(channel, aligned_vol.shape,
                                        dtype=aligned_vol.dtype, data=aligned_vol)

        except Exception as e:
            logger.error(f"Error during alignment for Code: {code}, ROI: {fov}, Error: {e}, {traceback.format_exc()}")
            raise


def volumetric_alignment(args: Args,
                         code_fov_pairs: Optional[List[Tuple[int, int]]] = None,
                         parallel_processes: int = 1,
                         bg_sub: Optional[str] = '',
                         accelerated: Optional[bool] = False,
                         **kwargs) -> None:
    r"""
    Coordinates the alignment of volumetric data across different fields of view (FOV) and codes.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param code_fov_pairs: A list of tuples, where each tuple is a (code, fov) pair. If None, uses all code and fov pairs from the args.
    :type code_fov_pairs: List[Tuple[int, int]], optional
    :param parallel_processes: The number of processes to use for parallel processing. Default is 1, which means no parallel processing.
    :type parallel_processes: int, optional
    :param bg_sub: Specifies the background subtraction method to be used. Can be "rolling_ball" or "top_hat". If not provided, no background subtraction will be applied.
    :type bg_sub: str, optional
    :param accelerated: Use GPU acceleration if you set it up. Can be. If not provided, Default is False.
    :type accelerated: bool, optional
    :param kwargs: Additional parameters to customize the alignment process. The possible options include:

        **General Options:**

        - `downsample_factors` (tuple of int): Downsampling factors for each dimension (z, y, x). Default is (1, 1, 1).
        - `downsample_steps` (list of tuples): List of steps to perform on downscaled volumes. Each step is a tuple of (step_name, step_kwargs).
        - `full_size_steps` (list of tuples): List of steps to perform on full-size volumes. Each step is a tuple of (step_name, step_kwargs).
        - `run_downsample_steps` (bool): Whether to execute the downsampled alignment steps. Default is True.
        - `low` (float): Low percentile value for intensity normalization. Default is 1.0.
        - `high` (float): High percentile value for intensity normalization. Default is 99.0.
        
        **Alignment Pipeline Steps:**
        
        Steps passed via `downsample_steps` or `full_size_steps` must conform to the `alignment_pipeline` function in BigStream. Each step is a tuple of the form `(step_name, step_kwargs)`. The available steps are:

        - `'ransac'`: Runs feature-point-based affine alignment using RANSAC. 
        - `'rigid'`: Performs a rigid affine alignment (rotation and translation only). 
        - `'affine'`: Performs a full affine alignment (translation, rotation, scaling, and shearing).
        - `'deform'`: Runs deformable alignment using B-splines or other methods.


    For more details on step-specific arguments, refer to the [alignment_pipeline documentation](https://github.com/JaneliaSciComp/bigstream/blob/master/bigstream/align.py#L1319).

    **Example Usage:**

    1. **With Downsampling and Full-Size Alignment:**

    .. code-block:: python

        volumetric_alignment(
            args, parallel_processes=4, bg_sub='rolling_ball',
            downsample_factors=(2, 2, 2),
            downsample_steps=[
                ('ransac', {'blob_sizes': [5, 150], 'safeguard_exceptions': True})
            ],
            full_size_steps=[
                ('affine', {'metric': 'NCC', 'optimizer': 'Powell'})
            ],
            run_downsample_steps=True,
            low=30.0, high=99.0
        )

    2. **Full-Size Only Alignment:**

    .. code-block:: python

        volumetric_alignment(
            args, parallel_processes=4, bg_sub='rolling_ball',
            full_size_steps=[
                ('affine', {'metric': 'NCC', 'optimizer': 'Powell'})
            ],
            run_downsample_steps=False,
            low=30.0, high=99.0
        )
    """
    child_processes = []
    tasks_queue = multiprocessing.Queue()
    q_lock = multiprocessing.Lock()

    if not code_fov_pairs:
        logger.info(f"Generating Code-FOV pairs using args.codes: {args.codes} and args.fovs: {args.fovs}.")
        code_fov_pairs = [[code_val, fov_val]
                          for code_val in args.codes for fov_val in args.fovs]

    for code, fov in code_fov_pairs:
        tasks_queue.put((code, fov, bg_sub))

    for w in range(int(parallel_processes)):
        try:
            p = multiprocessing.Process(
                target=execute_volumetric_alignment_bigstream,
                args=(args, tasks_queue, q_lock, accelerated),
                kwargs=kwargs)

            child_processes.append(p)
            p.start()

        except Exception as e:
            logger.error(f"Error starting process. Error: {e}")

    for p in child_processes:
        try:
            p.join()
        except Exception as e:
            logger.error(f"Error joining process {p.name}: {e}")


