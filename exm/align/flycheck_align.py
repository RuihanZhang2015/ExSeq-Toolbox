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

from tqdm import tqdm
from exm.io.io import nd2ToVol, nd2ToSlice, nd2ToChunk


## TODO what does mode refers to:
def transform_ref_code(args, code_fov_pairs=None, mode="all"):
    r"""For each volume specified in code_fov_pairs, convert from an nd2 file to an array, then save into an .h5 file.
    Args:
        args (args.Args): configuration options.
        code_fov_pairs (list): A list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
        mode (str): channels to run, should be one of 'all' (all channels), '405' (just the reference channel) or '4' (all channels other than reference). Default: ``'all'``
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


def computeMinFlann(
    fix,
    move,
    k=1,
    flann_idx_kdtree=0,
    flann_trees=5,
    checks=50,
    sift_mask=None,
    flann_mask=False,
    ratio=0.75,
):
    r"""Compute the min distance and L2-norm of distance of nearest neighbors of the key points in a fixed volume and their matches
    in a moving volume.
    Args:
        fix (np.ndarray): fixed image volume
        move (np.ndarray): moving image volume
        k (int): number of nearest neighbors, works only for 1 right now
        flann_idx_kdtree (int): FLANN algorithm to be used
        trees (int): number of FLANN trees
        checks (int): number of checks to use in FLANN
        sift_mask (bool): mask to be used for computing SIFT feature descriptors
        flann_mask (bool): mask to be used for FLANN algorithm
        ratio (float): ratio to be used for FLANN mask
    """

    sift = cv.SIFT_create()
    index_params = dict(algorithm=flann_idx_kdtree, trees=flann_trees)
    if checks is not None:
        search_params = dict(checks=checks)  # or pass empty dictionary
    else:
        search_params = {}

    flann = cv.FlannBasedMatcher(index_params, search_params)

    _, desf = sift.detectAndCompute(fix.astype("uint8"), sift_mask)
    _, desm = sift.detectAndCompute(move.astype("uint8"), sift_mask)

    if desf is not None and desm is not None:
        matches = flann.knnMatch(desf, desm, k=k)
    else:
        return None

    dists = [pt[0].distance for pt in matches]
    dists = np.asarray(dists)
    if dists.size > 0:
        norm = np.linalg.norm(dists)
        min_dist = np.min(dists)
    else:
        return None

    return norm, min_dist


def mutualInformation(im1, im2, bins=20):
    r"""Compute the mutual information for a joint histogram created using images im1 and im2.
    Args:
        im1 (np.ndarray): First image to be taken under consideration while creating histogram
        im2 (np.ndarray): Second image to be taken under consideration while creating histogram
        bins (int): number of sampling bins for the histogram
    """
    # create histogram
    hgram, _, _ = np.histogram2d(im1.ravel(), im2.ravel(), bins=bins)
    # convert bin counts to probability values
    pxy = hgram / float(np.sum(hgram))
    # compute marginal probabilities
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[:, None]
    # consider only non-zeros for total sum
    nzs = px_py > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def computeMaxMI(fix, move, min_array, z_min=0, num_minima=5):
    r"""For a given (fixed) z-slice, a moving volume and an array of minima, find the z-index having
    maximum mutual information.
    Args:
        fix (np.ndarray): fixed volume slice
        move (np.ndarray): entire moving volume (NOTE: this is not a z-slice!)
        min_array (np.ndarray): an array of local minima
        z_min (int): starting z-index in the moving volume
        num_minima (int): number of local minima to consider while checking maximum mutual information
    """
    min_sorted = np.sort(min_array)
    mi_result = dict()

    for row, min_val in enumerate(min_sorted[:num_minima]):
        z_ind = np.argwhere(min_array == min_val)[0][0] + z_min
        move_slice = move[z_ind, :, :]
        mi = mutualInformation(fix, move_slice)
        mi_result[z_ind] = mi

    max_mi_ind = max(mi_result, key=mi_result.get)
    max_mi = max(mi_result.values())

    return max_mi, max_mi_ind


def computeOffset(args, code_fov_pairs=None, path=None):
    r"""Calculates the z-offset between the fixed and moving volume and writes it to a .pkl.
    Args:
        args (args.Args): configuration options.
        code_fov_pairs (list): A list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
        path (string): path to save the dictionary. Default: ``None``
    """
    if not code_fov_pairs:
        code_fov_pairs = [
            [code, fov]
            for code in args.codes
            if code != args.ref_code
            for fov in args.fovs
        ]

    if not path:
        path = os.path.join(args.project_path, "processed/compute_offset")
        if not os.path.exists(path):
            os.makedirs(path)

    offset_dict = {}
    for code, fov in code_fov_pairs:
        # read image volumes
        fixed_vol = nd2ToVol(
            args.nd2_path.format(args.ref_code, "405", 4), fov, "405 SD"
        )
        mov_vol = nd2ToVol(args.nd2_path.format(code, "405", 4), fov, "405 SD")

        fix_slice = fixed_vol[0, :, :]
        dists = []

        for z in tqdm(
            mov_vol[
                30:180,
                :,
            ],
            desc="Computing FLANN distance...",
        ):
            try:
                _, dist = computeMinFlann(fix_slice, z)
                dists.append(dist)
            except:
                continue  # takes care of no matching keypoints

        _, max_ind = computeMaxMI(fix_slice, mov_vol, dists, z_min=30)
        # log in offset dictionary
        # note that the keys are strings instead of tuples (JSON rquires this)
        offset_dict.update({str(f"({code}, {fov})"): [max_ind]})

    # write offset dictionary to path
    with open(f"{path}/z_offset.json", "w") as f:
        json_object = json.dumps(offset_dict, indent=4)
        f.write(json_object)


def identify_matching_z(args, code_fov_pairs=None, path=None):
    r"""For each volume specified in code_fov_pairs, save a series of images that allow the user to match corresponding z-slices.
    Args:
        args (args.Args): configuration options.
        code_fov_pairs (list): A list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
        path (string): Path to save the images. Default: ``None``
    """
    import matplotlib.pyplot as plt

    if not code_fov_pairs:
        code_fov_pairs = [
            [code, fov]
            for code in args.codes
            if code != args.ref_code
            for fov in args.fovs
        ]

    if not path:
        path = os.path.join(args.project_path, "processed/align_matching_z")

    for code, fov in code_fov_pairs:

        if not os.path.exists(f"{path}/code{code}"):
            os.makedirs(f"{path}/code{code}")

        fig, axs = plt.subplots(2, 5, figsize=(25, 10))

        for i, z in enumerate(np.linspace(0, 200, 5)):

            im = nd2ToSlice(
                args.nd2_path.format(args.ref_code, "405", 4), fov, int(z), "405 SD"
            )

            axs[0, i].imshow(im, vmax=600)
            axs[0, i].set_xlabel(z)
            axs[0, i].set_title(f"Ref fov{fov} code{code}")

        for i, z in enumerate(np.linspace(0, 200, 5)):

            im = nd2ToSlice(args.nd2_path.format(code, "405", 4), fov, int(z), "405 SD")

            axs[1, i].imshow(im, vmax=600)
            axs[1, i].set_xlabel(z)
            axs[1, i].set_title(f"fov{fov} code{code}")

        plt.savefig(f"{path}/code{code}/fov{fov}.jpg")
        plt.close()


def correlation_lags(args, code_fov_pairs=None, path=None):
    r"""Calculates the z-offset between the fixed and moving volume and writes it to a .pkl. A returned offset of -x means that the fixed volume
    starts x slices before the move. A returned offset of x means that the fixed volume starts x slices after
    the move.
    Args:
        args (args.Args): configuration options.
        code_fov_pairs (list): A list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
        path (string): path to save the dictionary. Default: ``None``
    """
    from scipy import signal

    if not code_fov_pairs:
        code_fov_pairs = [
            [code, fov]
            for code in args.codes
            if code != args.ref_code
            for fov in args.fovs
        ]

    if not path:
        path = os.path.join(args.project_path, "processed/correlation_lags")
        if not os.path.exists(path):
            os.makedirs(path)

    lag_dict = {}
    for code, fov in code_fov_pairs:

        fixed_vol = nd2ToVol(
            args.nd2_path.format(args.ref_code, "405", 4), fov, "405 SD"
        )
        mov_vol = nd2ToVol(args.nd2_path.format(code, "405", 4), fov, "405 SD")

        intensities_fixed = np.array([np.mean(im.flatten()) for im in fixed_vol])
        intensities_mov = np.array([np.mean(im.flatten()) for im in mov_vol])

        correlation = signal.correlate(intensities_fixed, intensities_mov, mode="full")
        lags = signal.correlation_lags(
            intensities_fixed.size, intensities_mov.size, mode="full"
        )
        lag = lags[np.argmax(correlation)]

        if lag > 0:
            lag_dict[f"({code}, {fov})"] = [0, lag]

        else:
            lag_dict[f"({code}, {fov})"] = [abs(lag), 0]

    with open(f"{path}/z_offset.pkl", "wb") as f:
        pickle.dump(lag_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def align_truncated(args, code_fov_pairs=None):
    r"""For each volume in code_fov_pairs, find corresponding reference volume, truncate, then perform alignment.
    Args:
        args (args.Args): configuration options.
        code_fov_pairs (list): A list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
    """

    import SimpleITK as sitk

    sitk.ProcessObject_SetGlobalWarningDisplay(False)

    for code, fov in code_fov_pairs:

        if tuple([code, fov]) not in args.align_init:
            continue
        print(f"align_truncated: code{code},fov{fov}")

        # Get the indexes in the matching slices in two dataset
        fix_start, mov_start, last = args.align_init[tuple([code, fov])]

        # Fixed volume
        fix_vol = nd2ToChunk(
            args.nd2_path.format(args.ref_code, "405", 4),
            fov,
            fix_start,
            fix_start + last,
        )

        # Move volume
        mov_vol = nd2ToChunk(
            args.nd2_path.format(code, "405", 4), fov, mov_start, mov_start + last
        )

        # temp dicectory for the log files
        tmpdir_obj = tempfile.TemporaryDirectory()

        # Align
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetLogToFile(False)
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.SetOutputDirectory(tmpdir_obj.name)

        ## WE SHOULD MOVE SETTING THE PARAMETERS OUTSIDE OF THIS FUNCTION
        fix_vol_sitk = sitk.GetImageFromArray(fix_vol)
        fix_vol_sitk.SetSpacing(args.spacing)
        elastixImageFilter.SetFixedImage(fix_vol_sitk)

        mov_vol_sitk = sitk.GetImageFromArray(mov_vol)
        mov_vol_sitk.SetSpacing(args.spacing)
        elastixImageFilter.SetMovingImage(mov_vol_sitk)

        parameter_map = sitk.GetDefaultParameterMap("rigid")
        parameter_map["NumberOfSamplesForExactGradient"] = [
            "1000"
        ]  # NumberOfSamplesForExactGradient
        parameter_map["MaximumNumberOfIterations"] = [
            "15000"
        ]  # MaximumNumberOfIterations
        parameter_map["MaximumNumberOfSamplingAttempts"] = [
            "100"
        ]  # MaximumNumberOfSamplingAttempts
        parameter_map["FinalBSplineInterpolationOrder"] = [
            "1"
        ]  # FinalBSplineInterpolationOrder
        parameter_map["NumberOfResolutions"] = ["2"]
        elastixImageFilter.SetParameterMap(parameter_map)
        elastixImageFilter.Execute()

        transform_map = elastixImageFilter.GetTransformParameterMap()
        sitk.WriteParameterFile(transform_map[0], args.tform_path.format(code, fov))

        # Apply transform
        transform_map = sitk.ReadParameterFile(args.tform_path.format(code, fov))
        transformix = sitk.TransformixImageFilter()
        transformix.SetLogToFile(False)
        transformix.SetLogToConsole(False)
        transformix.SetTransformParameterMap(transform_map)

        # Just visualize the first 100 slices
        mov_vol_sitk = mov_vol_sitk[:, :, :100]

        transformix.SetMovingImage(mov_vol_sitk)
        transformix.Execute()
        out = sitk.GetArrayFromImage(transformix.GetResultImage())

        # Save the results
        with h5py.File(args.h5_path_cropped.format(code, fov), "w") as f:
            f.create_dataset("405", out.shape, dtype=out.dtype, data=out)

        tmpdir_obj.cleanup()


def inspect_align_truncated(args, fov_code_pairs=None, path=None):
    r"""For each volume in code_fov_pairs, save a series of images that allow the user to check the quality of alignmentt.
    Args:
        args (args.Args): configuration options.
        code_fov_pairs (list): A list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
        path (string): Path to save the images. Default: ``None``
    """

    import matplotlib.pyplot as plt

    for code, fov in fov_code_pairs:

        if tuple([code, fov]) not in args.align_init:
            continue
        print(f"inspect_align_truncated: code{code},fov{fov}")

        if not path:
            path = os.path.join(args.project_path, "processed/inspect_align_truncated")
            if not os.path.exists(path):
                os.makedirs(path)

        if not os.path.exists(f"{path}/code{code}"):
            os.makedirs(f"{path}/code{code}")

        fix_start, mov_start, last = args.align_init[tuple([code, fov])]
        z_stacks = np.linspace(fix_start, fix_start + last - 1, 5)

        # ---------- Full resolution -----------------
        fig, axs = plt.subplots(2, 5, figsize=(20, 5))

        for i, z in enumerate(z_stacks):
            im = nd2ToSlice(
                args.nd2_path.format(args.ref_code, "405", 4), fov, int(z), "405 SD"
            )
            axs[0, i].imshow(im, vmax=600)
            axs[0, i].set_xlabel(z)
            axs[0, i].set_ylabel("fix")

        for i, z in enumerate(z_stacks):
            with h5py.File(args.h5_path_cropped.format(code, fov), "r") as f:
                im = f["405"][int(z), :, :]
                im = np.squeeze(im)
            axs[1, i].imshow(im, vmax=600)
            axs[1, i].set_xlabel(z)
            axs[1, i].set_ylabel("transformed")
        plt.savefig(f"{path}/code{code}/fov{fov}_large.jpg")
        plt.close()

        # ------------ Top left corner-------------------
        fig, axs = plt.subplots(2, 5, figsize=(20, 5))
        for i, z in enumerate(z_stacks):
            im = nd2ToSlice(
                args.nd2_path.format(args.ref_code, "405", 4), fov, int(z), "405 SD"
            )[:300, :300]
            axs[0, i].imshow(im, vmax=600)
            axs[0, i].set_xlabel(z)
            axs[0, i].set_ylabel("fix")

        for i, z in enumerate(z_stacks):
            with h5py.File(args.h5_path_cropped.format(code, fov), "r") as f:
                im = f["405"][int(z), :300, :300]
                im = np.squeeze(im)
            axs[1, i].imshow(im, vmax=600)
            axs[1, i].set_xlabel(z)
            axs[1, i].set_ylabel("transformed")
        plt.savefig(f"{path}/code{code}/fov{fov}_topleft.jpg")
        plt.close()

        # ------------ Bottom right corner----------
        fig, axs = plt.subplots(2, 5, figsize=(20, 5))
        for i, z in enumerate(z_stacks):
            im = nd2ToSlice(
                args.nd2_path.format(args.ref_code, "405", 4), fov, int(z), "405 SD"
            )[1700:, 1700:]
            axs[0, i].imshow(im, vmax=600)
            axs[0, i].set_xlabel(z)
            axs[0, i].set_ylabel("fix")

        for i, z in enumerate(z_stacks):
            with h5py.File(args.h5_path_cropped.format(code, fov), "r") as f:
                im = f["405"][int(z), 1700:, 1700:]
                im = np.squeeze(im)
            axs[1, i].imshow(im, vmax=600)
            axs[1, i].set_xlabel(z)
            axs[1, i].set_ylabel("transformed")
        plt.savefig(f"{path}/code{code}/fov{fov}_bottomright.jpg")
        plt.close()


# TODO limit itk multithreading
# TODO add basic alignment approach
def transform_other_function(args, tasks_queue=None, q_lock=None, mode="all"):
    r"""Takes the transform found from the reference round and applies it to the other channels.
    Args:
        args (args.Args): configuration options.
        tasks_queue (list): A list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
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

            if tuple([code, fov]) not in args.align_init:
                continue
            print(f"transform_other_function: code{code},fov{fov}")

            # Load the start position
            fix_start, mov_start, last = args.align_init[tuple([code, fov])]

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

                # Change the size
                transform_map["Size"] = tuple([str(x) for x in mov_vol.shape[::-1]])

                # Shift the start
                trans_um = np.array(
                    [float(x) for x in transform_map["TransformParameters"]]
                )
                trans_um[-1] -= (fix_start - mov_start) * 4
                transform_map["TransformParameters"] = tuple([str(x) for x in trans_um])

                # Center of rotation
                cen_um = np.array(
                    [float(x) for x in transform_map["CenterOfRotationPoint"]]
                )
                cen_um[-1] += mov_start * 4
                transform_map["CenterOfRotationPoint"] = tuple([str(x) for x in cen_um])

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


def transform_other_code(args, code_fov_pairs=None, num_cpu=None, mode="all"):

    r"""Parallel processing support for transform_other_function.
    Args:
        args (args.Args): configuration options.
        code_fov_pairs (list): A list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
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
