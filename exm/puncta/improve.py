"""Enhancing Puncta Detection and Gene Correspondence."""

import tqdm
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from exm.utils import retrieve_all_puncta,retrieve_one_puncta, retrieve_vol
from exm.utils import configure_logger
logger = configure_logger('ExSeq-Toolbox')


def find_nearest_puncta(args, position, fov, code, center_dist=15, distance_threshold=8,GPU= False):
    r"""Reads in the locations of the puncta from a specified fov and code, then uses distance thresholding to consolidate (remove duplicate puncta) across channels. Can optionally utilize GPU acceleration.

    :param args: Configuration options.
    :param tuple position: (d0, d1, d2) coordinates of the position.
    :param int fov: Field of view.
    :param int code: the code of the volume chunk to be searched.
    :param int center_dist: Determines the size of the ROI around the input position in the d1 and d2 directions. Default is 15.
    :param int distance_threshold: Maximum allowed Euclidean distance from the input position to a maximum point. Default is 8.
    :param bool GPU: If True, GPU acceleration will be used. Default is False.

    :returns: A dictionary where the keys are channel names ('c0' to 'c3') and 'intensity', 'color', 'position'. Each channel key maps to another dictionary containing 'position', 'intensity', and 'distance' of the closest maximum point in this channel. 'intensity' key maps to a list of intensities for all channels. 'color' key maps to the channel with the maximum intensity. 'position' key maps to the position of the maximum point in the 'color' channel.If no maximum points were found in any channel, returns an empty dictionary.
    """
    if GPU:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter
        from cucim.skimage.feature import peak_local_max 

    # Input positions
    d0, d1, d2 = position
    d0, d1, d2 = int(d0), int(d1), int(d2)
    point_cloud1 = np.asarray([position])

    ROI_min = [max(0, d0-10), max(0, d1 - center_dist),
               max(0, d2 - center_dist)]
    ROI_max = [d0+10, d1 + center_dist, d2 + center_dist]

    new_puncta = {}
    for channel in range(4):

        # Search for local maximum
        vol = retrieve_vol(args, fov, code, channel, ROI_min, ROI_max)
        if GPU:
            vol = cp.array(vol)
        blurred = gaussian_filter(vol, 1, mode='reflect', cval=0)
        coords = peak_local_max(blurred, min_distance=7,
                                threshold_abs=110, exclude_border=False)
        if len(coords) == 0:
            continue
        if GPU:
            point_cloud2 = coords.get() + ROI_min
        else:
            point_cloud2 = coords + ROI_min

        # Find the nearest point
        temp1 = np.copy(point_cloud1)
        temp1[:, 0] = temp1[:, 0] * 0.5

        temp2 = np.copy(point_cloud2)
        temp2[:, 0] = temp2[:, 0] * 0.5

        distance = cdist(temp1, temp2, 'euclidean')

        # Find the index of the closest puncta from cloud 2 for each puncta in cloud 1
        index1 = np.argmin(distance, axis=1)[0]

        # Update the distance
        if distance[0, index1] > distance_threshold:
            continue

        new_position = point_cloud2[index1]
        new_puncta['c{}'.format(channel)] = {
            'position': np.array([new_position[0], new_position[1], new_position[2]]),
            'intensity': float(vol[new_position[0]-ROI_min[0], new_position[1]-ROI_min[1], new_position[2]-ROI_min[2]]),
            'distance': distance[0, index1],
        }

    if not new_puncta:
        return new_puncta

    new_puncta['intensity'] = [new_puncta['c{}'.format(
        c)]['intensity'] if 'c{}'.format(c) in new_puncta else 0 for c in range(4)]
    new_puncta['color'] = np.argmax(new_puncta['intensity'])
    new_puncta['position'] = new_puncta['c{}'.format(
        new_puncta['color'])]['position']

    return new_puncta


def puncta_all_nearest_points(args, puncta):
    r"""This function is used to find the nearest puncta for all missing codes in the given puncta's barcode. It iteratively calls the 'find_nearest_puncta' function for each missing code to find the corresponding nearest puncta.
    :param args.Args args: configuration options.
    :param dict puncta: Dictionary containing information about a puncta, including its 'barcode', 'position', and 'fov' values after consolidate codes.
    :returns: A dictionary where the keys are the names of the missing codes ('code0' to 'codeN'). Each key maps to another dictionary containing the information ('position', 'intensity', 'distance', and 'ref_code') of the nearest puncta found for that missing code. If no nearest puncta were found for any missing code, returns an empty dictionary.
    """
    # Find the missing codes
    barcode = puncta['barcode']
    missing_code_list = np.where(np.array(list(barcode)) == '_')[0]

    # Set the ref code
    def find_ref_code():
        for i in args.codes:
            if i not in missing_code_list:
                return i

    ref_code = find_ref_code()

    # For each missing code, search for local maximum, and find the closest point
    nearest_puncta_list = {}
    for code in missing_code_list:
        nearest_puncta = find_nearest_puncta(
            args, puncta['position'], puncta['fov'], code)
        if not nearest_puncta:
            continue
        nearest_puncta_list['code{}'.format(code)] = nearest_puncta
        nearest_puncta_list['code{}'.format(code)]['ref_code'] = ref_code

    return nearest_puncta_list


def improve_nearest(args, fov, num_missing_code=4):
    r"""This function retrieves all puncta within a specified field of view (fov), and then refines each puncta's information by adding the nearest puncta for all missing codes in the puncta's barcode. The refined puncta information is then saved to a pickle file in a subdirectory corresponding to the given fov.
    :param args.Args args: configuration options.
    :param int fov: Field of view.
    :param int num_missing_code: Number of missing codes in the barcode to allow a puncta to be processed. Default is 4.
    :returns: No return value. The function saves the improved puncta list to a pickle file named 'nearest_improved_puncta.pickle' in a subdirectory named 'fov{fov}' under the directory specified by args.work_path.
    """
    puncta_list = retrieve_all_puncta(args, fov)
    new_puncta_list = []

    for puncta in tqdm.tqdm(puncta_list):

        if puncta['barcode'].count('_') > num_missing_code:
            continue

        puncta['fov'] = fov

        nearest_puncta_list = puncta_all_nearest_points(args, puncta)

        puncta.update(nearest_puncta_list)

        s = ''
        for code in range(7):
            if 'code{}'.format(code) in puncta:
                s += str(puncta['code{}'.format(code)]['color'])
            else:
                s += '_'
        puncta['barcode'] = s

        new_puncta_list.append(puncta)

    with open(args.work_path + '/fov{}/improved_puncta_results.pickle'.format(fov), 'wb') as f:
        pickle.dump(new_puncta_list, f)


# TODO Merge with find_nearest_puncta
def find_nearest_points(point_cloud1,point_cloud2,distance_threshold=14):
    r"""Finds the nearest points between two point clouds based on Euclidean distance.
    This function computes the Euclidean distance between each point in `point_cloud1` and `point_cloud2`. 
    It then finds the nearest point in `point_cloud2` for each point in `point_cloud1`, within the specified 
    `distance_threshold`.
    :param np.ndarray point_cloud1: An Nx3 numpy array representing the first point cloud. Each row is a point, and the columns represent the x, y, and z coordinates respectively.
    :param np.ndarray point_cloud2: An Mx3 numpy array representing the second point cloud. Each row is a point, and the columns represent the x, y, and z coordinates respectively.
    :param float distance_threshold: The maximum allowed distance for points to be considered as 'nearest'. Points in `point_cloud2` that are farther away from a point in `point_cloud1` than this threshold will not be considered as 'nearest'. Default is 14.
    :returns: A list of dictionaries. Each dictionary represents a pair of nearest points and contains two keys: 'point0' and 'point1'. 'point0' is the index of a point in `point_cloud1`, and 'point1' is the index of its nearest point in `point_cloud2` within the `distance_threshold`.
    """
    import numpy as np
    from scipy.spatial.distance import cdist
    temp1 = np.copy(point_cloud1)
    temp1[:,0] = temp1[:,0] * 0.5
    temp2 = np.copy(point_cloud2)
    temp2[:,0] = temp2[:,0] * 0.5

    # Calculate euclidean distance between the two cloud points (point_cloud1 x point_cloud2)
    distance = cdist(temp1, temp2, 'euclidean')

    # Find the index of the closest puncta from cloud 2 for each puncta in cloud 1 
    index1 = np.argmin(distance, axis = 1)

    # Filter closest puncta pairs based on a set threshold
    pairs = [{'point0':i,'point1':index1[i]} for i in range(len(index1)) if distance[i,index1[i]] < distance_threshold]

    return pairs

# TODO Merge with puncta_all_nearest_points remove mute
def puncta_nearest_points(args,fov,puncta_index,search_code, verbose = False):
    r"""Identifies and retrieves the nearest puncta points based on the provided puncta index and search code. 
    The function finds the reference code, generates two point clouds (one for the original puncta and 
    another for potential new puncta points), and identifies the closest pairs between these two point clouds. 
    This process is repeated to find the closest points to the newly identified puncta points. 

    :param args: Configuration options, including methods for retrieving puncta and calculating nearest points.
    :param int fov: The field of view (fov) to consider.
    :param int puncta_index: The index of the puncta to start the search from.
    :param  int search_code: The search code to use when finding new puncta points.
    :param bool verbose: If set to True, the function will not print progress messages. 
                 If set to False, the function will print progress messages.
    :returns: A tuple containing the reference code, the closest new puncta point, and the closest point 
              to the new puncta point.

    """
    import numpy as np
    import pickle
    import pprint

    puncta = retrieve_one_puncta(args, fov, puncta_index)
    barcode = puncta['barcode']
    missing_code_list = np.where(np.array(list(barcode)) == '_')[0]

    # Set the ref code
    def find_ref_code():
        for i in args.codes:
            if i not in missing_code_list:
                return i

    ref_code = find_ref_code()

    if verbose:
        logger.info('------------------------')
        logger.info('The oiginal point in code 0:')
        logger.info(pprint.pformat(puncta['code{}'.format(ref_code)]))
        logger.info('------------------------')

    point_cloud1 = np.asarray([puncta['position']])
      
    with open(args.work_path + '/fov{}/result_code{}.pkl'.format(fov,search_code), 'rb') as f:
        new = pickle.load(f)

    point_cloud2 = np.asarray([x['position'] for x in new]) # other rounds puncta  

    pairs = find_nearest_points(point_cloud1,point_cloud2,distance_threshold=20)
    if len(pairs) == 0:
        if verbose:
            logger.info('no nearest point')
        return ref_code, None, None
    
    if verbose:
        logger.info('the closest point to fov{}, puncta {} in code{} is:{}'.format(fov,puncta_index,search_code,pairs[0]['point1']))
        logger.info(pprint.pformat(new[pairs[0]['point1']]))
        logger.info('------------------------')
    new[pairs[0]['point1']]['code'] = search_code
    new_position = new[pairs[0]['point1']]

        
    point_cloud1 = np.asarray([new[pairs[0]['point1']]['position']])
    with open(args.work_path + '/fov{}/result_code{}.pkl'.format(fov,ref_code), 'rb') as f:
        new = pickle.load(f)
    point_cloud2 = np.asarray([x['position'] for x in new]) # other rounds puncta  

    pairs = find_nearest_points(point_cloud1,point_cloud2)
    if len(pairs) == 0:
        if verbose:
            logger.info('no nearest point')
        return ref_code, new_position, None
    if verbose:
        logger.info('the closest point of the new point in code{} is:{}'.format(ref_code, pairs[0]['point1']))
        logger.info(pprint.pformat(new[pairs[0]['point1']]))
    closest_position = new[pairs[0]['point1']]
    new[pairs[0]['point1']]['code'] = ref_code
    return ref_code, new_position, closest_position