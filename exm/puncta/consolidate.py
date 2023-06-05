"""Consolidates puncta across channels and codes."""

import os
import h5py
import pickle
import queue
import numpy as np
import multiprocessing
from multiprocessing import current_process, Lock, Process, Queue

from exm.utils import chmod


def consolidate_channels_function(args, fov, code):
    r"""Reads in the locations of the puncta from a specified fov and code, then uses distance thresholding to consolidate (remove duplicate puncta) across channels.

    :param args.Args args: configuration options.
    :param int fov: field of view.
    :param int code: the code of the volume chunk to be consildate.
    """

    from scipy.spatial.distance import cdist

    def find_matching_points(point_cloud1, point_cloud2, distance_threshold=8):

        temp1 = np.copy(point_cloud1)
        temp1[:, 0] = temp1[:, 0] * 0.5
        temp2 = np.copy(point_cloud2)
        temp2[:, 0] = temp2[:, 0] * 0.5

        # Calculate euclidean distance between the two cloud points (point_cloud1 x point_cloud2)
        distance = cdist(temp1, temp2, "euclidean")
        # Find the index of the closest puncta from cloud 2 for each puncta in cloud 1
        index1 = np.argmin(distance, axis=1)
        # Find the index of the closest puncta from cloud 1 for each puncta in cloud 2
        index2 = np.argmin(distance, axis=0)
        # Pick puncta index that closest to each other (cloud 1 <-> cloud 2)
        valid = [i for i, x in enumerate(index1) if index2[x] == i]
        # Filter closest puncta pairs based on a set threshold
        pairs = [
            {"point0": i, "point1": index1[i]}
            for i in valid
            if (distance[i, index1[i]] < distance_threshold)
        ]

        return pairs

    print("Consolidate channels: code{}, fov{}".format(fov, code))

    # Open the coord total .pkl for the particular code and FOV
    with open(
        args.work_path + "/fov{}/coords_total_code{}.pkl".format(fov, code), "rb"
    ) as f:
        coords_total = pickle.load(f)

    ### Set the puncta in channel 640 as reference
    reference = [
        {"position": position, "c0": {"index": i, "position": position}}
        for i, position in enumerate(coords_total["c0"])
    ]

    ### Other channels '594','561','488'
    for c in [1, 2, 3]:

        point_cloud1 = np.asarray(
            [x["position"] for x in reference]
        )  # Reference 640 channel puncta coordination
        point_cloud2 = np.asarray(
            coords_total["c{}".format(c)]
        )  # other channels puncta coordination
        pairs = find_matching_points(
            point_cloud1, point_cloud2
        )  # find closest pairs of puncta between 2 channels
        # Write the matching pair for the other channels to the reference
        for pair in pairs:
            reference[pair["point0"]]["c{}".format(c)] = {
                "index": pair["point1"],
                "position": point_cloud2[pair["point1"]],
            }
        # Append the puncta without close pair to the 640 to the reference dict
        others = set(range(len(point_cloud2))) - set([pair["point1"] for pair in pairs])
        for other in others:
            reference.append(
                {
                    "position": point_cloud2[other],
                    "c{}".format(c): {"index": other, "position": point_cloud2[other]},
                }
            )

    # Get the index of the puncta point in the .h5 channel dataset
    with h5py.File(args.h5_path.format(code, fov), "r") as f:
        for i, duplet in enumerate(reference):
            temp = [
                f[args.channel_names[c]][tuple(duplet["c{}".format(c)]["position"])]
                if "c{}".format(c) in duplet
                else 0
                for c in range(4)
            ]
            duplet["color"] = np.argmax(
                temp
            )  # Channel of the puncta with highest intensity
            duplet[
                "intensity"
            ] = temp  # Intensity on different channels ['640','594','561','488']
            duplet["index"] = i  # puncta index
            duplet["position"] = duplet["c{}".format(duplet["color"])][
                "position"
            ]  # postion of the highest intensity puncta

    with open(args.work_path + "/fov{}/result_code{}.pkl".format(fov, code), "wb") as f:
        pickle.dump(reference, f)

    if args.permission:
        chmod(os.path.join(args.work_path, "fov{}/result_code{}.pkl".format(fov, code)))


def consolidate_channels(args, code_fov_pairs, num_cpu=None):
    r"""Wrapper around consolidate_channels_function to enable parallel processing.

    :param args.Args args: configuration options.
    :param list code_fov_pairs: a list of tuples, where each tuple is a (code, fov) pair. Default: ``None``
    :param int num_cpu: number of CPUs to use for processing. Default: ``None``
    """

    def run(tasks_queue, q_lock):

        while True:  # Check for remaining task in the Queue
            try:
                with q_lock:
                    fov, code = tasks_queue.get_nowait()
                    print(
                        "Remaining tasks to process : {}\n".format(tasks_queue.qsize())
                    )
            except queue.Empty:
                break
            else:
                consolidate_channels_function(args, fov, code)
                print(
                    "Consolidate Channels: Fov{}, Code{} Finished\n".format(fov, code)
                )

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
    tasks_queue = Queue()
    # Queue lock to avoid race condition.
    q_lock = Lock()

    # Add all the transform_other_channels to the queue.
    for code, fov in code_fov_pairs:
        tasks_queue.put((fov, code))

    for cpu_cores in range(int(cpu_execution_core)):
        p = Process(target=run, args=(tasks_queue, q_lock))
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()


def consolidate_codes_function(args, fov):
    r"""Reads in the locations of the puncta from a specified fov, then uses distance thresholding to consolidate (remove duplicate puncta) across codes.

    :param args.Args args: configuration options.
    :param int fov: field of view.
    """

    from scipy.spatial.distance import cdist

    def find_matching_points(point_cloud1, point_cloud2, distance_threshold=14):

        temp1 = np.copy(point_cloud1)
        temp1[:, 0] = temp1[:, 0] * 0.5
        temp2 = np.copy(point_cloud2)
        temp2[:, 0] = temp2[:, 0] * 0.5
        # Calculate euclidean distance between the two cloud points (point_cloud1 x point_cloud2)
        distance = cdist(temp1, temp2, "euclidean")
        # Find the index of the closest puncta from cloud 2 for each puncta in cloud 1
        index1 = np.argmin(distance, axis=1)
        # Find the index of the closest puncta from cloud 1 for each puncta in cloud 2
        index2 = np.argmin(distance, axis=0)
        # Pick puncta index that closest to each other (cloud 1 <-> cloud 2)
        valid = [i for i, x in enumerate(index1) if index2[x] == i]
        # Filter closest puncta pairs based on a set threshold
        pairs = [
            {"point0": i, "point1": index1[i]}
            for i in valid
            if distance[i, index1[i]] < distance_threshold
        ]

        return pairs

    ## get the consolidate_channels results for code 0
    code = args.ref_code
    with open(args.work_path + "/fov{}/result_code{}.pkl".format(fov, code), "rb") as f:
        new = pickle.load(f)

    # create reference using the code0 highest intensity puncta position and other details
    reference = [{"position": x["position"], "code0": x} for x in new]
    # Run through the remaining rounds
    for code in set(args.codes) - set([args.ref_code]):
        ## open other rounds consolidate_channels results
        with open(
            args.work_path + "/fov{}/result_code{}.pkl".format(fov, code), "rb"
        ) as f:
            new = pickle.load(f)

        # create
        point_cloud1 = np.asarray(
            [x["position"] for x in reference]
        )  # Reference puncta for code0 and other rounds
        point_cloud2 = np.asarray([x["position"] for x in new])  # other rounds puncta

        pairs = find_matching_points(
            point_cloud1, point_cloud2
        )  # find closest pairs of puncta between 2 rounds

        # Write the matching pair for other rounds to the reference
        for pair in pairs:
            reference[pair["point0"]]["code{}".format(code)] = new[pair["point1"]]

        # Append non-matching pairs for other rounds to the reference
        others = set(range(len(point_cloud2))) - set([pair["point1"] for pair in pairs])
        for other in others:
            reference.append(
                {"position": point_cloud2[other], "code{}".format(code): new[other]}
            )
    # index puncta from all channels
    reference = [{**x, "index": i} for i, x in enumerate(reference)]

    reference = [
        {
            **x,
            "barcode": "".join(
                [
                    str(x["code{}".format(code)]["color"])
                    if "code{}".format(code) in x
                    else "_"
                    for code in args.codes
                ]
            ),
        }
        for x in reference
    ]

    with open(args.work_path + "/fov{}/result.pkl".format(fov), "wb") as f:
        pickle.dump(reference, f)

    if args.permission:
        chmod(os.path.join(args.work_path, "fov{}/result.pkl".format(fov)))


def consolidate_codes(args, fov_list, num_cpu=None):
    r"""Wrapper around consolidate_codes_function to enable parallel processing.

    :param args.Args args: configuration options.
    :param list fov_list: a list of integers, where each integer is a field of view to process.
    :param int num_cpu: number of CPUs to use for processing. Default: ``None``
    """

    def run(tasks_queue, q_lock):

        while True:  # Check for remaining task in the Queue
            try:
                with q_lock:
                    fov = tasks_queue.get_nowait()
                    print(
                        "Remaining tasks to process : {}\n".format(tasks_queue.qsize())
                    )
            except queue.Empty:
                break
            else:
                consolidate_codes_function(args, fov)
                print("Consolidate Codes: fov{} Finished".format(fov))

    # Use a quarter of the available CPU resources to finish the tasks; you can increase this if the server is accessible for this task only.
    if num_cpu == None:
        if len(fov_list) < multiprocessing.cpu_count() / 4:
            cpu_execution_core = len(fov_list)
        else:
            cpu_execution_core = multiprocessing.cpu_count() / 4
    else:
        cpu_execution_core = num_cpu

    # List to hold the child processes.
    child_processes = []
    # Queue to hold all the puncta extraction tasks.
    tasks_queue = Queue()
    # Queue lock to avoid race condition.
    q_lock = Lock()

    # Add all the transform_other_channels to the queue.
    for fov in fov_list:
        tasks_queue.put((fov))

    for cpu_cores in range(int(cpu_execution_core)):
        p = Process(target=run, args=(tasks_queue, q_lock))
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()
