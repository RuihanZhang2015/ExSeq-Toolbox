import os
import pickle
import h5py
import random
import pandas as pd
import numpy as np
from pathlib import Path

from IPython.display import display
from PIL import Image

from skimage.restoration import rolling_ball
from skimage.morphology import disk
from scipy.ndimage import white_tophat
from scipy.stats import rankdata


from typing import Type, Optional

from exm.args import Args
from exm.utils.log import configure_logger
logger = configure_logger('ExSeq-Toolbox')


def chmod(path: Path) -> None:
    """
    Sets permissions so that users and the owner can read, write and execute files at the given path.

    :param path: Path in which privileges should be granted.
    :type path: pathlib.Path
    """
    if os.name != "nt":  # Skip for Windows OS
        try:
            path.chmod(0o766)  # octal notation for permissions
        except Exception as e:
            logger.error(
                f"Failed to change permissions for {path}. Error: {e}")
            raise


def retrieve_all_puncta(args, fov):
    r"""Returns all identified puncta for a given field of view.

    :param args.Args args: configuration options.
    :param int fov: field of view to return
    """
    with open(args.puncta_path + "/fov{}/result.pkl".format(fov), "rb") as f:
        return pickle.load(f)


def retrieve_one_puncta(args, fov, puncta_index):
    r"""Returns information about a single puncta, given a specified field of view and index.

    :param args.Args args: configuration options.
    :param int fov: field of view.
    :param int puncta_index: index of the puncta of interest
    """
    return retrieve_all_puncta(args, fov)[puncta_index]


def retrieve_img(args, fov, code, channel, ROI_min, ROI_max):
    r"""Returns the middle slice of a specified volume chunk.

    :param args.Args: configuration options.
    :param int fov: the field of fiew of the volume slice to be returned.
    :param int code: the code of the volume slice to be returned.
    :param int channel: the channel of the volume slice to be returned.
    :param list ROI_min: minimum coordinates of the volume chunk to take the middle slice of. Expects coordinates in the format of :math:`[z, y, x]`.
    :param list ROI_max: maximum coordinates of the volume chunk to take the middle slice of. Expects coordinates in the format of :math:`[z, y, x]`.
    """

    if ROI_min != ROI_max:
        zz = int((ROI_min[0] + ROI_max[0]) // 2)

    with h5py.File(args.h5_path.format(code, fov), "r") as f:
        im = f[args.channel_names[channel]][
            zz,
            max(0, ROI_min[1]): min(2048, ROI_max[1]),
            max(0, ROI_min[2]): min(2048, ROI_max[2]),
        ]
        im = np.squeeze(im)

    return im


def retrieve_vol(args, fov, code, c, ROI_min, ROI_max):
    r"""Returns a specified volume chunk.

    :param args.Args args: configuration options.
    :param int fov: the field of fiew of the volume chunk to be returned.
    :param int code: the code of the volume chunk to be returned.
    :param int channel: the channel of the volume chunk to be returned.
    :param list ROI_min: minimum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    :param list ROI_max: maximum coordinates of the volume chunk. Expects coordinates in the format of :math:`[z, y, x]`.
    """
    with h5py.File(args.h5_path.format(code, fov), "r") as f:
        vol = f[args.channel_names[c]][
            max(0, ROI_min[0]): ROI_max[0],
            max(0, ROI_min[1]): min(2048, ROI_max[1]),
            max(0, ROI_min[2]): min(2048, ROI_max[2]),
        ]
    return vol


def gene_barcode_mapping(args):
    r"""This function loads a CSV file `args.gene_digit_csv` containing gene symbols and corresponding barcodes. It converts the barcodes to digit representations and creates two mappings: 'digit2gene' (from digit representation to gene symbol) and 'gene2digit' (from gene symbol to digit representation). These mappings are useful for identifying genes associated with puncta barcode in a field of view.
    :param args.Args args: configuration options, including the path to the gene-digit CSV file and the mapping from code to number.
    :returns: A tuple containing three elements. The first element is a pandas DataFrame containing the original CSV data with an additional column for the digit representations of the barcodes. The second element is a dictionary mapping from digit representation to gene symbol ('digit2gene'). The third element is a dictionary mapping from gene symbol to digit representation ('gene2digit').
    """
    df = pd.read_csv(args.gene_digit_csv)
    df['Digits'] = [''.join([args.code2num[c] for c in barcode])
                    for barcode in df['Barcode']]
    digit2gene, gene2digit = {}, {}
    for i, row in df.iterrows():
        digit2gene[row['Digits']] = row['Symbol']
        gene2digit[row['Symbol']] = row['Digits']
    return df, digit2gene, gene2digit


def display_img(img):
    r"""
    This function displays an image using the Image module from the Python Imaging Library (PIL). The function supports images of type boolean and other numpy data types. For boolean images, the function multiplies the image by 255 to create an 8-bit grayscale image. For non-boolean images, the function simply converts the image to an 8-bit grayscale image.
    :param numpy.ndarray img: The input image to display. This can be a boolean or non-boolean numpy array.
    """
    if img.dtype is np.dtype(bool):
        display(Image.fromarray((img).astype(np.uint8)))
    else:
        display(Image.fromarray(img))


def retrieve_digit(args, digit):
    """This function retrieves all puncta with a specified barcode (represented as a digit) across all provided fields of view (fov). For each puncta that matches the barcode, it appends the puncta (with added fov information) to a list.
    :param args.Args args:  Configuration options, including the list of fovs and the method to retrieve all puncta for a given fov.
    :param digit: The barcode to search for, represented as a digit.
    :returns: A list of all puncta across all fovs that match the specified barcode. Each puncta is represented as a dictionary containing puncta information and the fov it was found in.
    """
    puncta_lists = []
    for fov in args.fovs:
        result = retrieve_all_puncta(args, fov)
        for puncta in result:
            if puncta['barcode'] == digit:
                puncta_lists.append({
                    **puncta,
                    'fov': fov
                })

    return puncta_lists


def retrieve_summary(args):
    r"""This function retrieves a summary of all puncta for each field of view (fov) in the provided fovs list. The summary includes the total number of each barcode across all fovs as well as the count of each barcode in individual fovs. The function then saves this summary to a CSV file.
    :param args.Args args: configuration options, including the list of fovs, the method to retrieve all puncta for a given fov, and the work path where the summary CSV file will be saved.
    :returns: A pandas DataFrame containing the summary of barcodes. The DataFrame is indexed by barcode with columns for total count ('number') and count per fov (e.g., 'fov1', 'fov2', ...). The DataFrame is sorted by total count in descending order.
    """
    import tqdm
    from collections import defaultdict

    summary = defaultdict(lambda: defaultdict(int))
    for fov in tqdm.tqdm(args.fovs):
        result = retrieve_all_puncta(args, fov)
        for entry in result:
            summary['number'][entry['barcode']] += 1
            summary[f'fov{fov}'][entry['barcode']] += 1
    summary = pd.DataFrame(summary).fillna(0).astype(int)
    summary = summary.sort_values(by='number', ascending=False)
    summary.to_csv(os.path.join(args.puncta_path, 'digit_summary.csv'))
    return summary


def retrieve_complete(args):
    r"""This function retrieves a complete summary of barcodes that are present in both the gene-barcode mapping and the overall barcode summary. The function returns a DataFrame sorted by gene names and also writes this DataFrame to a CSV file.
    :param args.Args args: configuration options, including methods for gene-barcode mapping, retrieving barcode summary, and the work path where the summary CSV file will be saved.
    :returns: A pandas DataFrame containing the complete summary of barcodes. The DataFrame is indexed by barcode with columns for total count ('number') and count per fov (e.g., 'fov1', 'fov2', ...). Additionally, it contains a 'gene' column that maps each barcode to its corresponding gene. The DataFrame is sorted by gene names in ascending order.

    """
    df, digit2gene, gene2digit = gene_barcode_mapping(args)
    summary = retrieve_summary(args)
    complete = summary.loc[list(set(df['Digits']) & set(summary.index))]
    complete['gene'] = [digit2gene[digit] for digit in complete.index]
    complete = complete.sort_values('gene')
    complete.to_csv(os.path.join(args.puncta_path, 'gene_summary.csv'))

    return complete


def retrieve_gene(args, gene):
    r"""This function retrieves all puncta associated with a specific gene across all fields of view (fovs). It leverages a Hamming distance function to match the barcode of each puncta with the gene of interest, permitting a maximum of one mismatch. It also writes the gene-barcode mapping to a CSV file.
    :param args.Args args: configuration options, including methods for gene-barcode mapping, retrieving all puncta, and the work path where the gene-barcode CSV file will be saved.
    :param str gene: The gene of interest for which all corresponding puncta across all fovs will be retrieved.
    :returns: A list of dictionaries, each representing a puncta associated with the gene. Each dictionary includes the puncta's properties, as well as the fov in which it is found.
    """
    def within_hamming_distance(a, b):
        diff = 0
        for x, y in zip(a, b):
            if x != y:
                diff += 1
            if diff >= 2:
                return False
        return True

    df, digit2gene, gene2digit = gene_barcode_mapping(args)
    digit = gene2digit[gene]

    puncta_lists = []
    for fov in args.fovs:
        result = retrieve_all_puncta(args, fov)
        for puncta in result:
            if within_hamming_distance(puncta['barcode'], digit):
                puncta_lists.append({
                    **puncta,
                    'fov': fov
                })
    df.to_csv(os.path.join(args.puncta_path,
              'gene_{}_digit_map.csv'.format(gene)))
    return puncta_lists


def generate_debug_candidate(args, gene=None, fov=None, num_missing_code=1):
    r"""Generates a candidate puncta for debugging purposes. The function first randomly selects a gene and retrieves all corresponding puncta. It then filters the puncta based on the number of missing codes in their barcodes. Finally, it randomly selects one puncta from the filtered list.
    :param args.Args args: configuration options, including methods for gene-barcode mapping and retrieving all puncta.
    :param str gene: The gene of interest, if none is provided a gene is randomly selected.
    :param int fov: The field of view (fov) to consider. If none is provided, all fovs are considered.
    :param int num_missing_code: The number of missing codes in the barcode of the puncta to be retrieved, defaults is 1.
    :returns: A single randomly chosen puncta that satisfies all the criteria (matching gene, within fov, correct number of missing codes).
    """
    complete = retrieve_complete(args)

    if not gene:
        gene = complete['gene'].values[np.random.randint(0, len(complete))]

    puncta_lists = retrieve_gene(args, gene)

    if fov:
        logger.info('Studying gene {} in fov '.format(gene, fov))
        puncta_lists = [
            puncta for puncta in puncta_lists if puncta['fov'] == fov]
    else:
        logger.info('Studying gene {} in all fovs'.format(gene))
    logger.info('Total barcode that matches gene {} '.format(
        gene, len(puncta_lists)))

    puncta_lists = [puncta for puncta in puncta_lists if puncta['barcode'].count(
        '_') == num_missing_code]
    if len(puncta_lists) == 0:
        logger.info('Total barcode with {} missing codes: {}'.format(
            num_missing_code, 0))
        return generate_debug_candidate(args, gene=None, fov=None, num_missing_code=1)
    else:
        logger.info('Total barcode with {} missing codes: {}'.format(
            num_missing_code, len(puncta_lists)))

    random_index = random.randint(0, len(puncta_lists)-1)
    return puncta_lists[random_index]


def get_offsets(filename):
    r"""Given the filename for the BDV/H5 XML file, returns the stitching offset as a :math:`(N,3)` array in the :math:`(X,Y,Z)` order. Returned values are expressed in :math:`\mu m`.

    :param str filename: the file name of the ``BDV/H5`` XML file, produced by the Big Stitcher plugin of fiji.
    """
    import xml.etree.ElementTree

    tree = xml.etree.ElementTree.parse(filename)
    root = tree.getroot()
    vtrans = list()
    for registration_tag in root.findall("./ViewRegistrations/ViewRegistration"):
        tot_mat = np.eye(4, 4)
        for view_transform in registration_tag.findall("ViewTransform"):
            affine_transform = view_transform.find("affine")
            mat = np.array(
                [float(a)
                 for a in affine_transform.text.split(" ")] + [0, 0, 0, 1]
            ).reshape((4, 4))
            tot_mat = np.matmul(tot_mat, mat)
        vtrans.append(tot_mat)

    def transform_to_translate(m):
        m[0, :] = m[0, :] / m[0][0]
        m[1, :] = m[1, :] / m[1][1]
        m[2, :] = m[2, :] / m[2][2]
        return m[:-1, -1]

    trans = [transform_to_translate(vt).astype(np.int64) for vt in vtrans]
    return np.stack(trans)


def visualize_progress(args) -> None:
    r"""Visualizes the progress of the ExSeq ToolBox."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    try:
        result = np.zeros((len(args.fovs), len(args.codes)))
        annot = np.asarray(
            [["{},{}".format(fov, code) for code in args.codes]
             for fov in args.fovs]
        )
        for fov in args.fovs:
            for code_index, code in enumerate(args.codes):

                if os.path.exists(args.h5_path.format(code, fov)):
                    result[fov, code_index] = 1
                else:
                    continue

                if os.path.exists(
                    args.puncta_path +
                        "/fov{}/result_code{}.pkl".format(fov, code)
                ):
                    result[fov, code_index] = 4
                    continue

                if os.path.exists(args.puncta_path + '/fov{}/coords_total_code{}.pkl'.format(fov, code)):
                    result[fov, code_index] = 3
                    continue

                try:
                    with h5py.File(args.h5_path.format(code, fov), "r+") as f:
                        if set(f.keys()) == set(args.channel_names):
                            result[fov, code_index] = 2
                except:
                    pass

        fig, ax = plt.subplots(figsize=(7, 20))
        ax = sns.heatmap(result, annot=annot, fmt="", vmin=0, vmax=4)
        plt.show()
        logger.info(
            "1: 405 done, 2: all channels done, 3:puncta extracted 4:channel consolidated"
        )
    except Exception as e:
        logger.error(f"Failed to visualize progress. Error: {e}")
        raise

# Background subtraction


def subtract_background_rolling_ball(volume: np.ndarray,
                                     radius: int = 50,
                                     num_threads: Optional[int] = 40) -> np.ndarray:
    """
    Performs background subtraction on a volume image using the rolling ball method.

    :param volume: The input volume image.
    :type volume: np.ndarray
    :param radius: The radius of the rolling ball used for background subtraction. Default is 50.
    :type radius: int, optional
    :param num_threads: The number of threads to use for the rolling ball operation. Default is 40.
    :type num_threads: int, optional
    :return: The volume image after background subtraction.
    :rtype: np.ndarray
    """
    corrected_volume = np.empty_like(volume)
    logger.info(f"Rolling_ball background subtraction")
    try:
        for slice_index in range(volume.shape[0]):
            corrected_volume[slice_index] = volume[slice_index] - rolling_ball(
                volume[slice_index], radius=radius, num_threads=num_threads)

        return corrected_volume
    except Exception as e:
        logger.error(f"Error during rolling ball background subtraction: {e}")
        raise


def subtract_background_top_hat(volume: np.ndarray,
                                radius: int = 50,
                                use_gpu: Optional[bool] = True) -> np.ndarray:
    """
    Performs top-hat background subtraction on a volume image.

    :param volume: The input volume image.
    :type volume: np.ndarray
    :param radius: The radius of the disk structuring element used for top-hat transformation. Default is 50.
    :type radius: int, optional
    :param use_gpu: If True, uses GPU for computation (requires cupy). Default is False.
    :type use_gpu: bool, optional
    :return: The volume image after background subtraction.
    :rtype: np.ndarray
    """
    structuring_element = disk(radius)
    corrected_volume = np.empty_like(volume)
    logger.info(f"top-hat background subtraction")
    try:
        if use_gpu:
            from cupyx.scipy.ndimage import white_tophat
            import cupy as cp

        for i in range(volume.shape[0]):
            if use_gpu:
                corrected_volume[i] = cp.asnumpy(
                    white_tophat(
                        cp.asarray(volume[i]),
                        structure=cp.asarray(structuring_element)
                    )
                )
            else:
                corrected_volume[i] = white_tophat(
                    volume[i], structure=structuring_element)

        return corrected_volume
    except Exception as e:
        logger.error(f"Error during top-hat background subtraction: {e}")
        raise

