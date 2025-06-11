"""
Puncta extraction module facilitates the extraction of puncta—distinct fluorescent spots indicative of molecular targets—from expansion microscopy data. It provides functions for processing this data on both CPUs and GPUs, enabling flexible and high-throughput analysis. The extraction process identifies puncta coordinates and saves them for further analysis.
"""

import os
import h5py
import pickle
import queue
import numpy as np

import collections
from typing import List, Tuple
from pathlib import Path
from multiprocessing import current_process, Lock, Process, Queue
from exm.args import Args
from exm.utils import configure_logger
from exm.utils.utils import chmod

logger = configure_logger('ExSeq-Toolbox')


def calculate_coords_gpu(args: Args,
                         tasks_queue: Queue,
                         device: int,
                         lock: Lock,
                         queue_lock: Lock) -> None:
    r"""
    Extracts puncta from volumes included in the task queue and saves their locations to a .pkl file using GPU acceleration.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param tasks_queue: A queue of tasks, where each task is a (code, fov) pair.
    :type tasks_queue: Queue[Tuple[int, int]]
    :param device: GPU device ID.
    :type device: int
    :param lock: A multiprocessing lock instance to avoid race condition when processes accessing the GPU.
    :type lock: Lock
    :param queue_lock: A multiprocessing lock instance to avoid race condition when processes accessing the task queue.
    :type queue_lock: Lock
    """
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
    from cucim.skimage.feature import peak_local_max

    chunk_size = 100  # TODO: Consider making this variable

    with cp.cuda.Device(device):
        while True:  # Check for remaining tasks in the Queues
            try:
                with queue_lock:
                    fov, code = tasks_queue.get_nowait()
                    logger.info(
                        f'Remaining tasks to process: {tasks_queue.qsize()}')
            except queue.Empty:
                logger.info(f"No tasks left for {current_process().name}")
                break

            coords_total = collections.defaultdict(list)

            try:
                with h5py.File(args.h5_path.format(code, fov), "r") as f:
                    num_z = len(f[args.channel_names[0]][:, 0, 0])

                for c in range(4):
                    for chunk in range((num_z // chunk_size) + 1):
                        with h5py.File(args.h5_path.format(code, fov), "r") as f:
                            img = cp.array(f[args.channel_names[c]][max(
                                chunk_size * chunk - 7, 0):min(chunk_size * (chunk + 1) + 7, num_z), :, :])

                        with lock:
                            gaussian_filter(img, 1, output=img,
                                            mode='reflect', cval=0)
                            coords = cp.array(peak_local_max(
                                img, min_distance=7, threshold_abs=args.thresholds[c], exclude_border=False).get())
                            coords[:, 0] += max(chunk_size * chunk - 7, 0)

                            if chunk == 0:
                                coords_total[f'c{c}'] = coords
                            else:
                                coords_total[f'c{c}'] = cp.concatenate(
                                    (coords_total[f'c{c}'], coords), axis=0)

                            del img, coords
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()

                for c in range(4):
                    coords_total[f'c{c}'] = np.unique(
                        coords_total[f'c{c}'].get(), axis=0)

                with open(args.puncta_path + f'fov{fov}/coords_total_code{code}.pkl', 'wb') as f:
                    pickle.dump(coords_total, f)

                if args.permission:
                    chmod(Path(args.puncta_path).joinpath(
                        f"fov{fov}/coords_total_code{code}.pkl"))

                logger.info(
                    f'Extract Puncta:  Code: {code} fov: {fov} Finished on {current_process().name}')

            except Exception as e:
                logger.error(
                    f"Error during puncta extraction for Fov:{fov}, Code:{code}. Error: {e}")
                raise


def puncta_extraction_gpu(args: Args,
                          tasks_queue: Queue,
                          num_gpu: int) -> None:
    r"""
    Wrapper around calculate_coords_gpu to enable parallel processing on GPU. 

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param tasks_queue: A queue of tasks, where each task is a (code, fov) pair.
    :type tasks_queue: Queue[Tuple[int, int]]
    :param num_gpu: Number of GPUs to use for processing.
    :type num_gpu: int
    """

    child_processes = []
    q_lock = Lock()

    logger.info(f'Total tasks to process: {tasks_queue.qsize()}')

    gpu_locks = [(i, Lock()) for i in range(num_gpu)]

    process_per_gpu = 1
    for gpu_id, gpu_lock in gpu_locks:
        for _ in range(process_per_gpu):
            try:
                p = Process(target=calculate_coords_gpu, args=(
                    args, tasks_queue, gpu_id, gpu_lock, q_lock))
                child_processes.append(p)
                p.start()
            except Exception as e:
                logger.error(
                    f"Error during puncta extraction on GPU {gpu_id}. Error: {e}")
                raise

    for p in child_processes:
        try:
            p.join()
        except Exception as e:
            logger.error(f"Error during joining GPU process. Error: {e}")
            raise

    logger.info("Puncta extraction on GPU completed successfully.")


def calculate_coords_cpu(args: Args,
                         tasks_queue: Queue,
                         queue_lock: Lock) -> None:
    r"""
    Extracts puncta from volumes included in the task queue using CPU and saves their locations to a .pkl file.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param tasks_queue: A queue of tasks, where each task is a (code, fov) pair.
    :type tasks_queue: Queue[Tuple[int, int]]
    :param queue_lock: Lock for the shared tasks queue to avoid race conditions.
    :type queue_lock: Lock
    """
    from scipy.ndimage import gaussian_filter
    from skimage.feature import peak_local_max

    chunk_size = 100

    while True:  # Check for remaining tasks in the Queues
        try:
            with queue_lock:
                fov, code = tasks_queue.get_nowait()
                logger.info(
                    f'Remaining tasks to process: {tasks_queue.qsize()}')
        except queue.Empty:
            logger.info(f"No tasks left for {current_process().name}")
            break

        else:
            logger.info(
                f'calculate_coords_cpu: Code: {code}, FOV: {fov} on {current_process().name}')

            coords_total = collections.defaultdict(list)

            with h5py.File(args.h5_path.format(code, fov), "r") as f:
                num_z = len(f[args.channel_names[0]][:, 0, 0])

            for c in range(4):
                for chunk in range((num_z // chunk_size) + 1):
                    with h5py.File(args.h5_path.format(code, fov), "r") as f:
                        img = f[args.channel_names[c]][max(
                            chunk_size * chunk - 7, 0):min(chunk_size * (chunk + 1) + 7, num_z), :, :]
                        f.close()

                    gaussian_filter(img, 1, output=img, mode='reflect', cval=0)
                    coords = peak_local_max(
                        img, min_distance=7, threshold_abs=args.thresholds[c], exclude_border=False)
                    coords[:, 0] += max(chunk_size * chunk - 7, 0)

                    if chunk == 0 or len(coords_total[f'c{c}']) == 0:
                        coords_total[f'c{c}'] = coords
                    else:
                        coords_total[f'c{c}'] = np.concatenate(
                            (coords_total[f'c{c}'], coords), axis=0)

            for c in range(4):
                coords_total[f'c{c}'] = np.unique(
                    coords_total[f'c{c}'], axis=0)

            with open(args.puncta_path + f'/fov{fov}/coords_total_code{code}.pkl', 'wb') as f:
                pickle.dump(coords_total, f)

            if args.permission:
                chmod(Path(args.puncta_path).joinpath(
                    f"fov{fov}/coords_total_code{code}.pkl"))

        logger.info(
            f'Extract Puncta:  Code: {code} fov: {fov} Finished on {current_process().name}')


def puncta_extraction_cpu(args: Args,
                          tasks_queue: Queue,
                          num_cpu: int) -> None:
    r"""
    Wrapper around calculate_coords_cpu to enable parallel processing on the CPU.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param tasks_queue: A queue of tasks, where each task is a (code, fov) pair.
    :type tasks_queue: Queue
    :param num_cpu: Number of CPUs to use for processing.
    :type num_cpu: int
    """

    logger.info(
        f"Starting puncta extraction on CPU for {tasks_queue.qsize()} tasks.")

    child_processes = []
    q_lock = Lock()

    try:
        for _ in range(num_cpu):
            p = Process(target=calculate_coords_cpu,
                        args=(args, tasks_queue, q_lock))
            child_processes.append(p)
            p.start()

        for p in child_processes:
            p.join()

        logger.info("Puncta extraction on CPU completed successfully.")
    except Exception as e:
        logger.error(f"Error during puncta extraction on CPU. Error: {e}")
        raise


def extract(args: Args,
            code_fov_pairs: List[Tuple[int, int]],
            use_gpu: bool = False,
            num_gpu: int = 3,
            num_cpu: int = 3) -> None:
    r"""
    Runs the extraction process (calculate_coords_cpu or calculate_coords_gpu) 
    for all codes and FOVs specified in code_fov_pairs.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param code_fov_pairs: A list of tuples, where each tuple is a (code, fov) pair.
    :type code_fov_pairs: List[Tuple[int, int]]
    :param use_gpu: Whether or not to enable GPU processing. Default is False.
    :type use_gpu: bool
    :param num_gpu: Number of GPUs to use for processing. Default is 3.
    :type num_gpu: int
    :param num_cpu: Number of CPUs to use for processing. Default is None which means it will use a quarter of available CPUs.
    :type num_cpu: Optional[int]
    """

    # Ensure directories exist for the specified FOVs
    for _, fov in code_fov_pairs:
        fov_path = os.path.join(args.puncta_path, f'fov{fov}')
        if not os.path.exists(fov_path):
            os.makedirs(fov_path)

    # Queue to hold all the puncta extraction tasks.
    tasks_queue = Queue()

    # Add all the extraction tasks to the queue.
    for code, fov in code_fov_pairs:
        tasks_queue.put((fov, code))

    try:
        if use_gpu:
            logger.info(
                f"Starting puncta extraction using GPU for {len(code_fov_pairs)} pairs.")
            puncta_extraction_gpu(args, tasks_queue, num_gpu)
        else:
            logger.info(
                f"Starting puncta extraction using CPU for {len(code_fov_pairs)} pairs.")
            puncta_extraction_cpu(args, tasks_queue, num_cpu)
        logger.info("Puncta extraction completed successfully.")
    except Exception as e:
        logger.error(f"Error during puncta extraction. Error: {e}")
        raise
