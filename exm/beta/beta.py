import h5py
import numpy as np

from scipy.stats import rankdata

from exm.args import Args
from exm.utils.log import configure_logger
logger = configure_logger('ExSeq-Toolbox')


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
            with h5py.File(args.h5_path.format(code, fov), "r") as f:
                fix_volume = f[args.ref_channel][()]

            for channel in args.channel_names:
                if channel == args.ref_channel:
                    continue

                with h5py.File(args.h5_path.format(code, fov), "r") as f:
                    mov_volume = f[channel][()]

                spacing = [0.40, 0.1625, 0.1625]

                # define alignment steps
                rigid_kwargs = {
                    'alignment_spacing': 0.5,
                    'shrink_factors': (8, 4, 2, 1),
                    'smooth_sigmas': (1., 1., 1., 1.),
                    'optimizer_args': {
                        'learningRate': 0.25,
                        'minStep': 0.,
                        'numberOfIterations': 400,
                    },
                }

                rigid = affine_align(
                    fix_volume, mov_volume,
                    spacing, spacing,
                    rigid=True,
                    fix_mask=(fix_volume > 105).astype(np.uint8),
                    mov_mask=(mov_volume > 105).astype(np.uint8), **rigid_kwargs
                )

                np.savetxt(args.tform_path.format(
                    code, f"fov{fov}_{channel}_affine.mat"), rigid)

                # apply affine only
                aligned_vol = apply_transform(
                    fix_volume, mov_volume,
                    spacing, spacing,
                    transform_list=[rigid,],
                )

                with h5py.File(args.h5_path.format(code, fov), "a") as f:
                    aligned_channel = channel + '_align'
                    if aligned_channel in f.keys():
                        del f[aligned_channel]
                    f.create_dataset(
                        aligned_channel, aligned_vol.shape, dtype=aligned_vol.dtype, data=aligned_vol)


def algin_channels(args: Args,
                   code_fov_pairs,
                   parallel_processes: int = 1) -> None:

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
