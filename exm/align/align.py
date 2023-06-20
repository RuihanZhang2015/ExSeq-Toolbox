"""
Code for volumetric alignment. For "thick" volumes (volumes that have more than 400 slices), use the alignment functions that end in "truncated".
"""
import json
import h5py
import pickle
import tempfile
import numpy as np
import cv2 as cv
import os
import queue
import multiprocessing
import skimage
import scipy

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.signal import find_peaks

from exm.io.io import nd2ToVol, nd2ToSlice, nd2ToChunk
from exm.utils import chmod


def transform_ref_code(args, code_fov_pairs=None, mode="all"):
    r"""For each volume specified in code_fov_pairs, convert from an nd2 file to an array, then save into an .h5 file.

    :param args.Args args: configuration options.
    :param list code_fov_pairs: a list of tuples where each tuple is a (code, fov) pair. Default: ``None``
    :param str mode: channels to run, should be one of ``all`` (all channels), `405` (just the reference channel) or `4` (all channels other than reference). Default: ``'all'``
    """

    if not code_fov_pairs:
        code_fov_pairs = [[args.ref_code, fov] for fov in args.fovs]

    for code, fov in code_fov_pairs:
        print("transform_ref_code: code = {}, fov={}".format(code, fov))
        with h5py.File(args.h5_path.format(code, fov), "a") as f:
            for channel_name_ind, channel_name in enumerate(args.channel_names):
                if mode == "405" and "405" not in channel_name:
                    continue
                if mode == "four" and "405" in channel_name:
                    continue
                if channel_name in f.keys():
                    continue
                fix_vol = nd2ToVol(
                    args.nd2_path.format(code, channel_name, channel_name_ind),
                    fov,
                    channel_name,
                )
                f.create_dataset(
                    channel_name, fix_vol.shape, dtype=fix_vol.dtype, data=fix_vol
                )


def mask(img, padding=250, chunks=1, pos=None):
    r"""
    Given an image volume, this function returns a mask to use for registration. The mask is created by finding and filling bounding boxes around content using every (#z-slices/chunks) slices.

    :param np.ndarray img: The image volume to create a mask for.
    :type img: np.ndarray
    :param int padding: The amount of padding to add around the identified bounding box.
    :type padding: int
    :param int chunks: The number of slices to use for masking.
    :type chunks: int
    :param list pos: A list of two elements ([start, end]), which specify the starting and ending z-positions of volume content.
    :type pos: list

    :return: A mask of the same size as the input volume, with areas of interest marked.
    :rtype: np.ndarray
    """


    from segment_anything import build_sam, SamAutomaticMaskGenerator
    import cv2

    final_mask = np.zeros(img.shape)

    mask_generator = SamAutomaticMaskGenerator(model=build_sam(checkpoint="sam_vit_h_4b8939.pth"),
                                               points_per_side=32,
                                               points_per_batch=64)

    start, end = pos
    if pos:
        img = img[start:end, :, :]

    assert chunks > 0
    chunk_size = int(img.shape[0]/chunks)

    for i in range(chunks):
        beginning_chunk_slice = int(i * chunk_size)

        if i == (chunks - 1):
            end_chunk_slice = int(img.shape[0])
        else:
            end_chunk_slice = int((i + 1) * chunk_size)

        sl = cv2.cvtColor(img[int(
            (beginning_chunk_slice+end_chunk_slice)/2)], cv2.COLOR_GRAY2BGR).astype('uint8')
        masks = mask_generator.generate(sl)

        min_ = np.percentile([mask['area'] for mask in masks], 20)
        max_ = np.percentile([mask['area'] for mask in masks], 80)

        # Discard masks with extremely small areas (noise) and masks with extremely large areas (background)
        masks = [mask['segmentation']
                 for mask in masks if mask['area'] < max_ and mask['area'] > min_]

        print(start+beginning_chunk_slice, start+end_chunk_slice)
        print(len(masks))
        if len(masks) > 1:

            # Concatenate the masks into one image
            overlaid_masks = np.sum(np.stack(masks, axis=-1), axis=2)

            # Remove instance segmentation (convert to binary)
            overlaid_masks[overlaid_masks > 0] = 1

            # Find bounding box around identified objects
            coords = cv2.findNonZero(overlaid_masks)
            x, y, w, h = cv2.boundingRect(coords)

            padding = padding  # Amount of padding around the bounding box.

            # Fill bounding box + paddings with ones.
            bounding_box = cv2.rectangle(np.zeros(img[int((beginning_chunk_slice+end_chunk_slice)/2)].shape), (max(
                x-padding, 0), max(y-padding, 0)), (min(x+w+padding, 2048), min(y+h+padding, 2048)), (1, 1, 1), -1)

            if pos:
                final_mask[start+beginning_chunk_slice:start +
                           end_chunk_slice, :, :] = bounding_box

            else:
                final_mask[beginning_chunk_slice:end_chunk_slice,
                           :, :] = bounding_box

    return final_mask


def align(args, code_fov_pairs=None, masking_params=None, mode="405"):
    r"""For each volume in code_fov_pairs, find corresponding reference volume, then perform alignment.

    :param args.Args args: configuration options.
    :param list code_fov_pairs: a list of tuples, where each tuple is a ``(code, fov)`` pair. Default: ``None``
    :param list masking_params: list of params to use for masking ([[padding_fixed, chunks_fixed, [start_fixed, end_fixed]], [padding_mov, chunks_mov, [start_mov, end_mov]]]). If None, does not mask.  Default: ``None``
    :param str mode: whether to align just the anchoring channel ("405") or all channels ("all"). Default: ``False``
    """

    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalWarningDisplay(False)

    for code, fov in code_fov_pairs:

        print(f"align: code{code},fov{fov}")

        if not os.path.exists(os.path.join(args.processed_path, "code{}".format(code))):
            os.makedirs(os.path.join(
                args.processed_path, "code{}".format(code)))

        # Fixed volume
        fix_vol = nd2ToVol(args.nd2_path.format(args.ref_code, "405", 4), fov)

        # Move volume
        mov_vol = nd2ToVol(args.nd2_path.format(code, "405", 4), fov)
        
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
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.SetOutputDirectory(tmpdir_obj.name)
        
        elastixImageFilter.SetFixedImage(fix_vol_sitk)
        elastixImageFilter.SetMovingImage(mov_vol_sitk)

        # Translation across x, y, and z only
        parameter_map = sitk.GetDefaultParameterMap("translation")
        parameter_map["NumberOfSamplesForExactGradient"] = [
            "1000"]  # NumberOfSamplesForExactGradient
        parameter_map["MaximumNumberOfIterations"] = [
            "25000"
        ]  # MaximumNumberOfIterations
        parameter_map["MaximumNumberOfSamplingAttempts"] = [
            "2000"
        ]  # MaximumNumberOfSamplingAttempts
        parameter_map["FinalBSplineInterpolationOrder"] = [
            "1"
        ]  # FinalBSplineInterpolationOrder
        parameter_map["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"]
        parameter_map["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"]
        parameter_map["NumberOfResolutions"] = ["5"]
        parameter_map["FixedImagePyramidSchedule"] = [
            "10 10 10 8 8 8 4 4 4 2 2 2 1 1 1"
        ]
        elastixImageFilter.SetParameterMap(parameter_map)

        # Translation + rotation
        parameter_map = sitk.GetDefaultParameterMap("rigid")
        parameter_map["NumberOfSamplesForExactGradient"] = [
            "1000"
        ]  # NumberOfSamplesForExactGradient
        parameter_map["MaximumNumberOfIterations"] = [
            "25000"
        ]  # MaximumNumberOfIterations
        parameter_map["MaximumNumberOfSamplingAttempts"] = [
            "2000"
        ]  # MaximumNumberOfSamplingAttempts
        parameter_map["FinalBSplineInterpolationOrder"] = [
            "1"
        ]  # FinalBSplineInterpolationOrder
        parameter_map["FixedImagePyramid"] = ["FixedShrinkingImagePyramid"]
        parameter_map["MovingImagePyramid"] = ["MovingShrinkingImagePyramid"]
        parameter_map["NumberOfResolutions"] = ["1"]
        parameter_map["FixedImagePyramidSchedule"] = ["1 1 1"]
        parameter_map["MovingImagePyramidSchedule"] = ["1 1 1"]
        elastixImageFilter.AddParameterMap(parameter_map)

        # Translation, rotation, scaling and shearing
        parameter_map = sitk.GetDefaultParameterMap("affine")
        parameter_map["NumberOfSamplesForExactGradient"] = [
            "1000"
        ]  # NumberOfSamplesForExactGradient
        parameter_map["MaximumNumberOfIterations"] = [
            "25000"
        ]  # MaximumNumberOfIterations
        parameter_map["MaximumNumberOfSamplingAttempts"] = [
            "2000"
        ]  # MaximumNumberOfSamplingAttempts
        parameter_map["FinalBSplineInterpolationOrder"] = [
            "1"
        ]  # FinalBSplineInterpolationOrder
        parameter_map["FixedImagePyramid"] = ["FixedShrinkingImagePyramid"]
        parameter_map["MovingImagePyramid"] = ["MovingShrinkingImagePyramid"]
        parameter_map["NumberOfResolutions"] = ["1"]
        parameter_map["FixedImagePyramidSchedule"] = ["1 1 1"]
        parameter_map["MovingImagePyramidSchedule"] = ["1 1 1"]
        elastixImageFilter.AddParameterMap(parameter_map)

        if masking_params:

            padding, chunks, pos = masking_params[0]

            fix_mask = mask(fix_vol, padding, chunks, pos)
            fix_mask = sitk.GetImageFromArray(fix_mask.astype("uint8"))
            fix_mask.CopyInformation(fix_vol_sitk)
            elastixImageFilter.SetFixedMask(fix_mask)

            padding, chunks, pos = masking_params[1]

            move_mask = mask(mov_vol, padding, chunks, pos)
            move_mask = sitk.GetImageFromArray(move_mask.astype("uint8"))
            move_mask.CopyInformation(mov_vol_sitk)
            elastixImageFilter.SetMovingMask(move_mask)

        elastixImageFilter.Execute()

        transform_map = elastixImageFilter.GetTransformParameterMap()

        # sitk.WriteParameterFile(transform_map[0], args.tform_path.format(code, str(fov) + ".0"))
        # sitk.WriteParameterFile(transform_map[1], args.tform_path.format(code, str(fov) + ".1"))
        # sitk.WriteParameterFile(transform_map[2], args.tform_path.format(code, str(fov) + ".2"))

        if mode == "405":
            out = sitk.GetArrayFromImage(
                transformixImageFilter.GetResultImage())
            with h5py.File(args.h5_path.format(code, fov), "a") as f:
                if channel in ["405"]:
                    del f[channel]
                f.create_dataset(channel, out.shape, dtype=out.dtype, data=out)

        if mode == "all":
            for channel_ind, channel in enumerate(args.channel_names):

                print(channel)
                mov_vol = nd2ToVol(
                    args.nd2_path.format(
                        code, channel, channel_ind), fov, channel
                )
                mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
                mov_vol_sitk.SetSpacing(args.spacing)

                transformixImageFilter = sitk.TransformixImageFilter()
                transformixImageFilter.SetMovingImage(mov_vol_sitk)
                transformixImageFilter.SetTransformParameterMap(
                    elastixImageFilter.GetTransformParameterMap()
                )
                transformixImageFilter.LogToConsoleOn()
                transformixImageFilter.Execute()

                out = sitk.GetArrayFromImage(
                    transformixImageFilter.GetResultImage())
                with h5py.File(args.h5_path.format(code, fov), "a") as f:
                    if channel in f.keys():
                        del f[channel]
                    f.create_dataset(channel, out.shape,
                                     dtype=out.dtype, data=out)

        tmpdir_obj.cleanup()


def compute_gradient(img: np.ndarray):
    r"""Find the pixel gradient for each slice in the image volume. Return means and standard deviations.
    :param np.array img: volumetric image.
    """
    means, std_devs = [], []

    for im_slice in tqdm(img, desc='Computing laplacians...'):
        laplacian = cv.Laplacian(im_slice, cv.CV_64F)
        mean, std_dev = cv.meanStdDev(laplacian)
        means.append(mean)
        std_devs.append(std_dev)

    return means, std_devs


def plot_peaks(func_op: np.ndarray, height=28, distance=25):
    r"""Plot the pixel gradients.
    :param np.array func_op: array of pixel gradients to plot.
    :param int height: height of horizontal line across the plot.
    :param int distance: distance to allow between peaks. 
    """
    slice_idx = np.arange(len(func_op))
    reshape_op = np.array(func_op).reshape(-1,)
    # find local extrema
    peaks, _ = find_peaks(reshape_op, height=height, distance=distance)
    # plot stuff
    plt.plot(reshape_op)
    plt.plot(peaks, reshape_op[peaks], 'x')
    plt.plot(height*np.ones_like(reshape_op), "--", color="gray")
    plt.show()


def offset(std_dev: list, height: int, distance: int, debug_mode: bool):
    r"""Returns the offset of the image volume. Requires manual tuning for sparse volumes.
    :param np.array std_dev: array of standard deviations to plot.
    :param int height: height of horizontal line across the plot.
    :param int distance: distance to allow between peaks. 
    :param bool debug_mode: if True, generates plot that allows for debugging parameters.
    """
    reshape_op = np.array(std_dev).reshape(-1,)
    peaks, _ = find_peaks(reshape_op, height=height, distance=distance)
    if debug_mode:
        plot_peaks(std_dev, height, distance)

    start_idx, last_idx = peaks[1], peaks[-1]

    return start_idx, last_idx


'''
# TODO limit itk multithreading
# TODO add basic alignment approach
def transform_other_function(args, tasks_queue=None, q_lock=None, mode="all"):
    r"""Takes the transform found from the reference round and applies it to the other channels.
    Args:
        args (args.Args): configuration options. 
        tasks_queue (list): a list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
        q_lock (multiporcessing.Lock): a multiporcessing.Lock instance to avoid race condition when processes accessing the task_queue. Default: ``None``
        mode (str): channels to run, should be one of 'all' (all channels), '405' (just the reference channel) or '4' (all channels other than reference). Default: ``'all'``
    """

    import SimpleITK as sitk

    while True:  # Check for remaining task in the Queue

        try:
            with q_lock:
                fov, code = tasks_queue.get_nowait()
                print("Remaining tasks to process : {}".format(tasks_queue.qsize()))
        except queue.Empty:
            print("No task left for " + multiprocessing.current_process().name)
            break
        else:

            for channel_name_ind, channel_name in enumerate(args.channel_names):

                with h5py.File(args.h5_path.format(code, fov), "a") as f:

                    if mode == "405":
                        if channel_name != "405":
                            continue
                    elif mode == "four":
                        if channel_name == "405":
                            continue
                    if channel_name in f.keys():
                        continue

                # Load the moving volume
                mov_vol = nd2ToVol(
                    args.nd2_path.format(code, channel_name, channel_name_ind),
                    fov,
                    channel_name,
                )
                mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
                mov_vol_sitk.SetSpacing(args.spacing)

                # Read the transform map
                transform_map = sitk.ReadParameterFile(
                    args.tform_path.format(code, fov)
                )

                if "code{},fov{}".format(code, fov) in args.align_z_init:

                    fix_start, mov_start, last = args.align_init[
                        "code{},fov{}".format(code, fov)
                    ]
                    # Change the size
                    transform_map["Size"] = tuple([str(x) for x in mov_vol.shape[::-1]])

                    # Shift the start
                    trans_um = np.array(
                        [float(x) for x in transform_map["TransformParameters"]]
                    )
                    trans_um[-1] -= (fix_start - mov_start) * 4
                    transform_map["TransformParameters"] = tuple(
                        [str(x) for x in trans_um]
                    )

                    # Center of rotation
                    cen_um = np.array(
                        [float(x) for x in transform_map["CenterOfRotationPoint"]]
                    )
                    cen_um[-1] += mov_start * 4
                    transform_map["CenterOfRotationPoint"] = tuple(
                        [str(x) for x in cen_um]
                    )

                # Apply the transform
                transformix = sitk.TransformixImageFilter()
                transformix.SetTransformParameterMap(transform_map)
                transformix.SetMovingImage(mov_vol_sitk)
                transformix.SetLogToFile(False)
                transformix.SetLogToConsole(False)
                transformix.Execute()
                out = sitk.GetArrayFromImage(transformix.GetResultImage())

                with h5py.File(args.h5_path.format(code, fov), "a") as f:
                    f.create_dataset(channel_name, out.shape, dtype=out.dtype, data=out)
'''

'''
def transform_other_code(args, code_fov_pairs=None, num_cpu=None, mode="all"):

    r"""Parallel processing support for transform_other_function.
    Args:
        args (args.Args): configuration options.
        code_fov_pairs (list): a list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
        num_cpu (int): the number of cpus to use for parallel processing. Default: ``8``
        mode (str): channels to run, should be one of 'all' (all channels), '405' (just the reference channel) or '4' (all channels other than reference). Default: ``'all'``
    """

    os.environ["OMP_NUM_THREADS"] = "1"

    # Use a quarter of the available CPU resources to finish the tasks; you can increase this if the server is accessible for this task only.
    if num_cpu == None:
        if len(code_fov_pairs) < multiprocessing.cpu_count() / 4:
            cpu_execution_core = len(code_fov_pairs)
        else:
            cpu_execution_core = multiprocessing.cpu_count() / 4
    else:
        cpu_execution_core = num_cpu
    # List to hold the child processes.
    child_processes = []
    # Queue to hold all the puncta extraction tasks.
    tasks_queue = multiprocessing.Queue()
    # Queue lock to avoid race condition.
    q_lock = multiprocessing.Lock()
    # Get the extraction tasks starting time.

    # Clear the child processes list.
    child_processes = []

    # Add all the align405 to the queue.
    for code, fov in code_fov_pairs:
        tasks_queue.put((fov, code))

    for w in range(int(cpu_execution_core)):
        p = multiprocessing.Process(
            target=transform_other_function, args=(args, tasks_queue, q_lock, mode)
        )
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()
        
'''
