"""
Puncta consolidation module is designed to refine puncta detection by eliminating redundancies across different channels and codes. It processes the data to merge closely located puncta into a single entity based on a distance threshold, which helps in reducing noise and improving the accuracy of the puncta detection process.
"""
import os
import h5py
import pickle
import queue
import numpy as np
from scipy.spatial.distance import cdist

from pathlib import Path
from typing import List, Tuple, Optional
import multiprocessing
from multiprocessing import Lock, Process, Queue

from exm.args import Args
from exm.utils.log import configure_logger
from exm.utils.utils import chmod

logger = configure_logger('ExSeq-Toolbox')


def consolidate_channels_function(args: Args, tasks_queue: int, q_lock: multiprocessing.Lock) -> None:
    r"""
    Consolidates (removes duplicate puncta) across channels using distance thresholding, 
    called by `consolidate_channels`.

    :param args: Configuration options. This should be an instance of a configuration class or similar structure.
    :type args: Any
    :param tasks_queue: Queue containing tuples of field of view (FOV) and code to be processed.
    :type tasks_queue: multiprocessing.Queue[Tuple[int, int]]
    :param q_lock: Lock for ensuring process-safe operations on the tasks queue.
    :type q_lock: multiprocessing.Lock
    """
    while True:
            try:
                with q_lock:
                    fov, code = tasks_queue.get_nowait()
                    logger.info(
                        f"Remaining tasks to process: {tasks_queue.qsize()}")
            except queue.Empty:
                break
            else:
                try:
                    
                    def find_matching_points(point_cloud1, point_cloud2, distance_threshold=None):
                        if distance_threshold is None:
                            distance_threshold = getattr(args, 'consolidation_distance_threshold', 8.0)
                        if len(point_cloud1) == 0 or len(point_cloud2) == 0:
                            return []
                        temp1 = np.copy(point_cloud1)
                        temp1[:, 0] = temp1[:, 0] * 0.5
                        temp2 = np.copy(point_cloud2)
                        temp2[:, 0] = temp2[:, 0] * 0.5
                        distance = cdist(temp1, temp2, "euclidean")

                        index1 = np.argmin(distance, axis=1)
                        index2 = np.argmin(distance, axis=0)
                        valid = [i for i, x in enumerate(index1) if index2[x] == i]
                        pairs = [{"point0": i, "point1": index1[i]}
                                for i in valid if (distance[i, index1[i]] < distance_threshold)]
                        return pairs

                
                    logger.info(
                        f"Starting to consolidate channels for Code: {code}, FOV: {fov}")

                    with open(args.puncta_path + f"/fov{fov}/coords_total_code{code}.pkl", "rb") as f:
                        coords_total = pickle.load(f)

                    reference = [{"position": position, "c0": {"index": i, "position": position}}
                                for i, position in enumerate(coords_total["c0"])]

                    # Other channels
                    for c in [1, 2, 3]:

                        point_cloud1 = np.asarray([x["position"] for x in reference])
                        point_cloud2 = np.asarray(coords_total[f"c{c}"])
                        if point_cloud2.size != 0:
                            pairs = find_matching_points(point_cloud1, point_cloud2)
                        else:
                            continue

                        for pair in pairs:
                            reference[pair["point0"]][f"c{c}"] = {
                                "index": pair["point1"], "position": point_cloud2[pair["point1"]], }

                        others = set(range(len(point_cloud2))) - \
                            set([pair["point1"] for pair in pairs])
                        for other in others:
                            reference.append({"position": point_cloud2[other], f"c{c}": {
                                            "index": other, "position": point_cloud2[other]}, })

                    with h5py.File(args.h5_path.format(code, fov), "r") as f:
                        for i, duplet in enumerate(reference):
                            temp = [f[args.channel_names[c]][tuple(
                                duplet[f"c{c}"]["position"])] if f"c{c}" in duplet else 0 for c in range(4)]

                            duplet["color"] = np.argmax(temp)
                            duplet["intensity"] = temp  # Intensity on different channels ['640','594','561','488']
                            duplet["index"] = i  
                            duplet["position"] = duplet["c{}".format(duplet["color"])]["position"]  

                    with open(args.puncta_path + f"/fov{fov}/result_code{code}.pkl", "wb") as f:
                        pickle.dump(reference, f)

                    if args.permission:
                        chmod(Path(args.puncta_path).joinpath(
                            f"fov{fov}/result_code{code}.pkl"))

                except Exception as e:
                    logger.error(
                        f"Error during channels consolidation for Code: {code}, FOV: {fov}. Error: {e}")
                    raise


def consolidate_channels(args: Args,
                         code_fov_pairs: List[Tuple[int, int]],
                         num_cpu: Optional[int] = None) -> None:
    r"""
    Wrapper around `consolidate_channels_function` to enable parallel processing.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param code_fov_pairs: A list of tuples, where each tuple is a (code, fov) pair.
    :type code_fov_pairs: List[Tuple[int, int]]
    :param num_cpu: Number of CPUs to use for processing. If not provided, defaults to a quarter of the available CPUs.
    :type num_cpu: int, optional 
    """

    # Determine the number of CPU cores to use.
    if num_cpu is None:
        cpu_execution_core = min(
            len(code_fov_pairs), multiprocessing.cpu_count() // 4)
    else:
        cpu_execution_core = num_cpu

    child_processes = []
    tasks_queue = Queue()
    q_lock = Lock()

    # Add all the consolidation tasks to the queue.
    for code, fov in code_fov_pairs:
        tasks_queue.put((fov, code))

    for _ in range(cpu_execution_core):
        try:
            p = Process(target=consolidate_channels_function, args=(args,tasks_queue,q_lock))
            child_processes.append(p)
            p.start()
        except Exception as e:
            logger.error(f"Error starting process for CPU core. Error: {e}")
            raise

    for p in child_processes:
        p.join()


def consolidate_codes_function(args: Args, tasks_queue: int,q_lock: multiprocessing.Lock) -> None:
    r"""
    Reads in the locations of the puncta from a specified field of view (FOV), then uses distance thresholding 
    to consolidate (i.e., remove duplicate puncta) across codes.

    :param args: Configuration options. This should be an instance of a configuration class or similar structure.
    :type args: Any
    :param tasks_queue: Queue containing tasks to be processed.
    :type tasks_queue: Queue
    :param q_lock: Lock for ensuring process-safe operations on the tasks queue.
    :type q_lock: multiprocessing.Lock
    """
    while True:
            try:
                with q_lock:
                    fov = tasks_queue.get_nowait()
                    logger.info(
                        f"Remaining tasks to process: {tasks_queue.qsize()}")
            except queue.Empty:
                break
            else:
                try:

                    logger.info(
                        f"Consolidate Codes for fov {fov} finished successfully.")

                    def find_matching_points(point_cloud1, point_cloud2, distance_threshold=None):
                        if distance_threshold is None:
                            distance_threshold = getattr(args, 'consolidation_distance_threshold', 8.0)
                        if len(point_cloud1) == 0 or len(point_cloud2) == 0:
                            return []
                        temp1 = np.copy(point_cloud1)
                        temp1[:, 0] = temp1[:, 0] * 0.5
                        temp2 = np.copy(point_cloud2)
                        temp2[:, 0] = temp2[:, 0] * 0.5
                        distance = cdist(temp1, temp2, "euclidean")
                        index1 = np.argmin(distance, axis=1)
                        index2 = np.argmin(distance, axis=0)
                        valid = [i for i, x in enumerate(index1) if index2[x] == i]
                        pairs = [{"point0": i, "point1": index1[i]}
                                for i in valid if (distance[i, index1[i]] < distance_threshold)]
                        return pairs


                    with open(args.puncta_path + f"/fov{fov}/result_code{args.ref_code}.pkl", "rb") as f:
                        new = pickle.load(f)

                    reference = [{"position": x["position"], f"code{args.ref_code}": x} for x in new]

                    for code in set(args.codes) - set([args.ref_code]):
                        with open(args.puncta_path + f"/fov{fov}/result_code{code}.pkl", "rb") as f:
                            new = pickle.load(f)

                        point_cloud1 = np.asarray([x["position"] for x in reference])
                        point_cloud2 = np.asarray([x["position"] for x in new])

                        if len(point_cloud2) == 0 or len(point_cloud1) == 0:
                            logger.warning(f"Code {code} for FOV {fov} has no puncta.")
                            continue

                        pairs = find_matching_points(point_cloud1, point_cloud2)

                        for pair in pairs:
                            reference[pair["point0"]][f"code{code}"] = new[pair["point1"]]

                        others = set(range(len(point_cloud2))) - \
                            set([pair["point1"] for pair in pairs])
                        for other in others:
                            reference.append(
                                {"position": point_cloud2[other], f"code{code}": new[other]})

                    reference = [{**x, "index": i} for i, x in enumerate(reference)]
                    reference = [{**x, "barcode": "".join([str(x[f"code{code}"]["color"])
                                                        if f"code{code}" in x else "_" for code in args.codes])} for x in reference]

                    with open(args.puncta_path + f"/fov{fov}/result.pkl", "wb") as f:
                        pickle.dump(reference, f)

                    if args.permission:
                        chmod(Path(args.puncta_path).joinpath(f"fov{fov}/result.pkl"))

                    logger.info(
                        f"Consolidation of codes for FOV {fov} completed successfully.")

                except Exception as e:
                    logger.error(
                        f"Error during codes consolidation for FOV {fov}. Error: {e}")
                    raise

def consolidate_codes(args: Args, fov_list: List[int], num_cpu: Optional[int] = None) -> None:
    r"""
    Wrapper around consolidate_codes_function to enable parallel processing.

    :param args: Configuration options.
    :type args: Args
    :param fov_list: A list of integers, where each integer is a field of view to process.
    :type fov_list: List[int]
    :param num_cpu: Number of CPUs to use for processing. Default is None which means it will use a quarter of available CPUs.
    :type num_cpu: Optional[int]
    """

    # Determine the number of CPU cores to use.
    if num_cpu is None:
        cpu_execution_core = min(
            len(fov_list), multiprocessing.cpu_count() // 4)
    else:
        cpu_execution_core = num_cpu

    child_processes = []
    tasks_queue = Queue()
    q_lock = Lock()

    for fov in fov_list:
        tasks_queue.put(fov)

    for _ in range(int(cpu_execution_core)):
        p = Process(target=consolidate_codes_function, args=(args,tasks_queue, q_lock))
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()
