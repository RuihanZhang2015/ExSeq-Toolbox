"""
The Utils module within the ExSeq Toolbox provides a comprehensive suite of utility functions to support the preprocessing, retrieval, manipulation, and visualization of expansion microscopy data.
"""
import os
import pickle
import h5py
import random
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

from IPython.display import display
from PIL import Image

from skimage import exposure
from skimage.restoration import rolling_ball
from skimage.morphology import disk
from scipy.ndimage import white_tophat , zoom , median_filter
from scipy.stats import rankdata

from typing import Type, Optional, Dict, List, Tuple, Union

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
            path.chmod(0o777)  # octal notation for permissions
        except Exception as e:
            logger.error(
                f"Failed to change permissions for {path}. Error: {e}")
            raise


def retrieve_all_puncta(args, fov: int) -> List[Dict]:
    r"""
    Returns all identified puncta for a given field of view.

    This function loads and returns all puncta data from a pickle file for the specified field of view. The path to the
    pickle file is constructed using the configuration options provided in the `args` parameter.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param fov: The field of view for which to return all identified puncta.
    :type fov: int
    :return: The data of all puncta identified in the specified field of view.
    :rtype: List[Dict]  

    """
    try:
        puncta_file_path = f"{args.puncta_path}/fov{fov}/result.pkl"
        with open(puncta_file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Pickle file not found for fov {fov}: {e}")
    except pickle.UnpicklingError as e:
        logger.error(f"Error unpickling data for fov {fov}: {e}")


def retrieve_one_puncta(args, fov: int, puncta_index: int) -> Dict:
    r"""
    Retrieves information about a specific puncta from a given field of view.

    This function uses the provided configuration options to access and return
    data for a single puncta, identified by its index, within the specified field of view.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param fov: The field of view from which to retrieve the puncta.
    :type fov: int
    :param puncta_index: The index of the specific puncta to retrieve.
    :type puncta_index: int

    :return: A dictionary containing information about the puncta.
    :rtype: Dict
    """
    if not isinstance(fov, int):
        logger.error("The field of view (fov) must be an integer.")
        raise
    if not isinstance(puncta_index, int):
        logger.error("The puncta index must be an integer.")
        raise

    all_puncta = retrieve_all_puncta(args, fov)

    try:
        return all_puncta[puncta_index]
    except Exception as e:
        logger.error(f"The specified puncta_index is out of range {e}.")
        raise


def retrieve_img(args, fov: int, code: int, channel: int, ROI_min: List[int], ROI_max: List[int]) -> np.ndarray:
    r"""
    Returns the middle slice of a specified volume chunk.

    This function retrieves a middle z-slice from a 3D volume chunk specified by its field of view, code, and channel.
    The ROI (Region of Interest) is defined by minimum and maximum coordinates.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param fov: The field of view of the volume slice to be returned.
    :type fov: int
    :param code: The code of the volume slice to be returned.
    :type code: int
    :param channel: The channel of the volume slice to be returned.
    :type channel: int
    :param ROI_min: Minimum coordinates of the volume chunk in the format of [z, y, x].
    :type ROI_min: List[int]
    :param ROI_max: Maximum coordinates of the volume chunk in the format of [z, y, x].
    :type ROI_max: List[int]

    :return: A 2D numpy array representing the middle z-slice of the specified volume chunk.
    :rtype: np.ndarray

    """
    if not (len(ROI_min) == len(ROI_max) == 3):
        logger.error(
            "ROI_min and ROI_max must both be lists of three integers.")
        raise

    if ROI_min != ROI_max:
        zz = int((ROI_min[0] + ROI_max[0]) // 2)
    else:
        logger.error("ROI_min and ROI_max cannot be the same.")
        raise

    try:
        with h5py.File(args.h5_path.format(code, fov), "r") as f:
            im = f[args.channel_names[channel]][
                zz,
                max(0, ROI_min[1]): min(2048, ROI_max[1]),
                max(0, ROI_min[2]): min(2048, ROI_max[2]),
            ]
            im = np.squeeze(im)
    except Exception as e:
        logger.error(f"An error occurred while retrieving the image: {e}")
        raise

    return im


def retrieve_vol(args, fov: int, code: int, c: int, ROI_min: List[int], ROI_max: List[int]) -> np.ndarray:
    r"""
    Returns a specified volume chunk from a dataset.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param fov: The field of view of the volume chunk to be returned.
    :type fov: int
    :param code: The code of the volume chunk to be returned.
    :type code: int
    :param c: The channel of the volume chunk to be returned.
    :type c: int
    :param ROI_min: Minimum coordinates of the volume chunk in the format of [z, y, x].
    :type ROI_min: List[int]
    :param ROI_max: Maximum coordinates of the volume chunk in the format of [z, y, x].
    :type ROI_max: List[int]
    :return: A numpy array representing the retrieved volume chunk.
    :rtype: h5py.Dataset

    """

    try:
        # Access the HDF5 file and retrieve the specified volume
        with h5py.File(args.h5_path.format(code, fov), "r") as f:
            vol = f[args.channel_names[c]][
                max(0, ROI_min[0]): ROI_max[0],
                max(0, ROI_min[1]): min(2048, ROI_max[1]),
                max(0, ROI_min[2]): min(2048, ROI_max[2]),
            ]
        return vol
    except OSError as e:
        logger.error(f"Error accessing file: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid ROI coordinates: {e}")
        raise


def gene_barcode_mapping(args) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    r"""
    Loads a CSV file containing gene symbols and corresponding barcodes, and creates mappings between them.

    This function reads a CSV file specified by `args.gene_digit_csv`, which contains gene symbols and their
    corresponding barcodes. It converts the barcodes into digit representations and creates two mappings:
    'digit2gene' for mapping from digit representation to gene symbol, and 'gene2digit' for mapping from
    gene symbol to digit representation. These mappings are useful for identifying genes associated with
    puncta barcodes in a field of view.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :returns: A tuple containing:
                - A pandas DataFrame with the original CSV data and an additional column for digit representations.
                - A dictionary mapping from digit representation to gene symbol ('digit2gene').
                - A dictionary mapping from gene symbol to digit representation ('gene2digit').
    :rtype: Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(args.gene_digit_csv)
        # Convert barcodes to digit representations
        df['Digits'] = [''.join([args.code2num[c] for c in barcode])
                        for barcode in df['Barcode']]
        # Initialize the mappings
        digit2gene, gene2digit = {}, {}
        # Populate the mappings
        for i, row in df.iterrows():
            digit2gene[row['Digits']] = row['Symbol']
            gene2digit[row['Symbol']] = row['Digits']
        return df, digit2gene, gene2digit
    except FileNotFoundError as e:
        logger.error(f"The gene-digit CSV file could not be found: {e}")
        raise
    except KeyError as e:
        logger.error(f"Expected columns are missing from the CSV: {e}")
        raise


def display_img(img: Union[np.ndarray, bool]) -> None:
    r"""
    Displays an image using the Image module from the Python Imaging Library (PIL).

    The function supports images of type boolean and other numpy data types. For boolean images, the function
    multiplies the image by 255 to create an 8-bit grayscale image. For non-boolean images, the function simply
    converts the image to an 8-bit grayscale image without scaling.

    :param img: The input image to display. This can be a boolean or non-boolean numpy array.
    :type img: Union[np.ndarray, bool]
    """
    # Check if the image is of boolean type, if so, convert to 8-bit by multiplying by 255
    if img.dtype == bool:
        img_to_display = (img * 255).astype(np.uint8)
    else:
        # For non-boolean images, ensure the image is in 8-bit format
        img_to_display = img.astype(np.uint8)

    # Convert to a PIL image and display it
    display(Image.fromarray(img_to_display))


def retrieve_digit(args, digit: str) -> List[Dict]:
    r"""
    Retrieves all puncta with a specified barcode (represented as a digit) across all fields of view.

    This function iterates over all provided fields of view (FOVs) and retrieves puncta that match
    the specified barcode. Each matching puncta, along with its FOV information, is appended to a list.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param digit: The barcode to search for, represented as a digit.
    :type digit: str
    :returns: A list of dictionaries where each dictionary contains information about a puncta and the FOV it was found in.
    :rtype: List[Dict]
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


def retrieve_summary(args) -> pd.DataFrame:
    r"""
    Retrieves a summary of all puncta for each field of view (FOV).

    This function iterates over the provided list of FOVs, retrieves all puncta for each FOV, and aggregates
    the count of each barcode across all FOVs and individually per FOV. The summary is then saved to a CSV file.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :returns: A pandas DataFrame containing the summary of barcodes. The DataFrame is indexed by barcode with columns
              for total count ('number') and count per FOV (e.g., 'fov1', 'fov2', ...). The DataFrame is sorted by
              total count in descending order.
    :rtype: pd.DataFrame
    """
    try:
        summary = defaultdict(lambda: defaultdict(int))
        for fov in args.fovs:
            result = retrieve_all_puncta(args, fov)
            for entry in result:
                summary['number'][entry['barcode']] += 1
                summary[f'fov{fov}'][entry['barcode']] += 1
        summary_df = pd.DataFrame(summary).fillna(0).astype(int)
        summary_df = summary_df.sort_values(by='number', ascending=False)

        csv_path = os.path.join(args.puncta_path, 'digit_summary.csv')
        if not os.path.exists(os.path.dirname(csv_path)):
            raise FileNotFoundError(
                f"Directory does not exist for saving the CSV file: {os.path.dirname(csv_path)}")

        summary_df.to_csv(csv_path)
        return summary_df

    except Exception as e:
        logger.error(
            f"An error occurred while retrieving summary or saving the CSV file: {e}")
        raise


def retrieve_complete(args) -> pd.DataFrame:
    r"""
    Retrieves a complete summary of barcodes present in both the gene-barcode mapping and the overall barcode summary.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :returns: A pandas DataFrame containing the complete summary of barcodes, indexed by barcode with columns for
              total count ('number') and count per fov (e.g., 'fov1', 'fov2', ...), and a 'gene' column mapping
              each barcode to its corresponding gene. Sorted by gene names in ascending order.
    :rtype: pd.DataFrame
    """
    try:
        df, digit2gene, gene2digit = gene_barcode_mapping(args)
        summary = retrieve_summary(args)

        complete = summary.loc[list(set(df['Digits']) & set(summary.index))]
        complete['gene'] = [digit2gene[digit] for digit in complete.index]
        complete = complete.sort_values('gene')

        csv_path = os.path.join(args.puncta_path, 'gene_summary.csv')
        complete.to_csv(csv_path)

        return complete
    except Exception as e:
        logger.error(
            f"An error occurred during the retrieval process or CSV file saving: {e}")
        raise


def retrieve_gene(args, gene: str) -> List[Dict]:
    r"""
    Retrieves all puncta associated with a specific gene across all fields of view (FOVs).

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param gene: The gene of interest for which all corresponding puncta across all FOVs will be retrieved.
    :type gene: str
    :returns: A list of dictionaries, each representing a puncta associated with the gene, including puncta's properties
              and the FOV in which it is found.
    :rtype: List[Dict]
    """
    def within_hamming_distance(a: str, b: str) -> bool:
        """Check if two barcodes are within a Hamming distance of 1."""
        diff = sum(1 for x, y in zip(a, b) if x != y)
        return diff < 2

    try:
        df, digit2gene, gene2digit = gene_barcode_mapping(args)

        # Retrieve the barcode digit for the specified gene
        digit = gene2digit[gene]

        puncta_lists = []
        for fov in args.fovs:
            result = retrieve_all_puncta(args, fov)
            for puncta in result:
                if within_hamming_distance(str(puncta['barcode']), str(digit)):
                    puncta_lists.append({
                        **puncta,
                        'fov': fov
                    })

        # Save the gene-barcode mapping to a CSV file
        gene_csv_path = os.path.join(
            args.puncta_path, f'gene_{gene}_digit_map.csv')
        df.to_csv(gene_csv_path)

        return puncta_lists
    except KeyError as e:
        logger.error(f"Gene {gene} not found in gene2digit mapping: {e}")
        raise
    except Exception as e:
        logger.error(
            f"An error occurred during the retrieval process or CSV file saving: {e}")
        raise


def generate_debug_candidate(args, gene: Optional[str] = None, fov: Optional[int] = None, num_missing_code: int = 1) -> Optional[Dict]:
    """
    Generates a candidate puncta for debugging purposes.

    The function first randomly selects a gene if not provided and retrieves all corresponding puncta. It then filters
    the puncta based on the number of missing codes in their barcodes. Finally, it randomly selects one puncta from
    the filtered list.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param gene: The gene of interest, if none is provided a gene is randomly selected.
    :type gene: Optional[str]
    :param fov: The field of view (FOV) to consider. If none is provided, all FOVs are considered.
    :type fov: Optional[int]
    :param num_missing_code: The number of missing codes in the barcode of the puncta to be retrieved. Default is 1.
    :type num_missing_code: int
    :returns: A single randomly chosen puncta that satisfies all the criteria (matching gene, within FOV, correct
              number of missing codes).
    :rtype: Optional[Dict]
    """
    try:
        complete = retrieve_complete(args)

        # Randomly select a gene if not provided
        if not gene:
            gene = complete['gene'].sample().iloc[0]

        puncta_lists = retrieve_gene(args, gene)

        # Filter puncta by field of view if provided
        if fov is not None:
            logger.info(f'Studying gene {gene} in fov {fov}')
            puncta_lists = [
                puncta for puncta in puncta_lists if puncta['fov'] == fov]
        else:
            logger.info(f'Studying gene {gene} in all fovs')

        logger.info(
            f'Total barcode that matches gene {gene}: {len(puncta_lists)}')

        # Filter puncta by the number of missing codes
        puncta_lists = [puncta for puncta in puncta_lists if puncta['barcode'].count(
            '_') == num_missing_code]

        if len(puncta_lists) == 0:
            logger.info(
                f'Total barcode with {num_missing_code} missing codes: 0')
            return None  # Or raise an exception if the function should not return None
        else:
            logger.info(
                f'Total barcode with {num_missing_code} missing codes: {len(puncta_lists)}')

        # Randomly select and return a puncta from the filtered list
        random_index = random.randint(0, len(puncta_lists)-1)
        return puncta_lists[random_index]
    except ValueError as e:
        logger.error(
            f"An error occurred during debug candidate generation: {e}")
        raise


def get_offsets(filename: str) -> np.ndarray:
    r"""
    Given the filename for the BDV/H5 XML file, returns the stitching offset as an (N,3) array in (X,Y,Z) order.

    The offsets are expressed in micrometers (µm) and are extracted from the XML file produced by the Big Stitcher
    plugin of Fiji.

    :param filename: The file name of the BDV/H5 XML file.
    :type filename: str
    :return: An array of stitching offsets in the format of (X, Y, Z).
    :rtype: np.ndarray

    :raises FileNotFoundError: If the XML file cannot be found.
    :raises ET.ParseError: If there is an error parsing the XML file.
    :raises ValueError: If the XML file has an unexpected structure or if the affine transformation cannot be read.
    """
    try:
        # Parse the XML file
        tree = ET.parse(filename)
        root = tree.getroot()
        vtrans: List[np.ndarray] = []

        # Extract the view transformations
        for registration_tag in root.findall("./ViewRegistrations/ViewRegistration"):
            tot_mat = np.eye(4, 4)
            for view_transform in registration_tag.findall("ViewTransform"):
                affine_transform = view_transform.find("affine")
                if affine_transform is None or affine_transform.text is None:
                    raise ValueError(
                        "Affine transformation not found or is empty.")

                mat = np.array(
                    [float(a) for a in affine_transform.text.split(
                        " ")] + [0, 0, 0, 1]
                ).reshape((4, 4))
                tot_mat = np.matmul(tot_mat, mat)
            vtrans.append(tot_mat)

        # Define a function to convert transformation matrices to translation vectors
        def transform_to_translate(m: np.ndarray) -> np.ndarray:
            m[0, :] = m[0, :] / m[0][0]
            m[1, :] = m[1, :] / m[1][1]
            m[2, :] = m[2, :] / m[2][2]
            return m[:-1, -1]

        # Apply the transformation and stack the results
        trans = [transform_to_translate(vt).astype(np.int64) for vt in vtrans]
        return np.stack(trans)

    except FileNotFoundError as e:
        logger.error(f"The XML file could not be found: {e}")
        raise
    except ET.ParseError as e:
        logger.error(f"Error parsing the XML file: {e}")
        raise
    except ValueError as e:
        logger.error(f"Error processing the XML file: {e}")
        raise


def visualize_progress(args) -> None:
    r"""
    Visualizes the progress of the ExSeq Toolbox.

    This function creates a heatmap visualizing the completion status of different steps in the ExSeq Toolbox
    for each field of view (FOV) and each code.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args

    """
    try:
        # Initialize the result matrix with zeros
        result = np.zeros((len(args.fovs), len(args.codes)))
        # Create annotation for the heatmap
        annot = np.asarray(
            [["{},{}".format(fov, code) for code in args.codes]
             for fov in args.fovs]
        )

        # Check the progress for each FOV and code
        for fov_index, fov in enumerate(args.fovs):
            for code_index, code in enumerate(args.codes):
                h5_path = args.h5_path.format(code, fov)
                puncta_path = args.puncta_path
                result_code_path = f"{puncta_path}/fov{fov}/result_code{code}.pkl"
                coords_total_code_path = f"{puncta_path}/fov{fov}/coords_total_code{code}.pkl"

                if os.path.exists(h5_path):
                    result[fov_index, code_index] = 1

                    if os.path.exists(result_code_path):
                        result[fov_index, code_index] = 4
                    elif os.path.exists(coords_total_code_path):
                        result[fov_index, code_index] = 3
                    else:
                        try:
                            with h5py.File(h5_path, "r") as f:
                                if set(f.keys()) == set(args.channel_names):
                                    result[fov_index, code_index] = 2
                        except Exception as e:
                            logger.warning(
                                f"Could not read file {h5_path}: {e}")

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(7, 20))
        sns.heatmap(result, annot=annot, fmt="", vmin=0, vmax=4, ax=ax)
        plt.show()
        logger.info(
            "1: 405 done, 2: all channels done, 3: puncta extracted, 4: channel consolidated")
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
                from scipy.ndimage import white_tophat
                corrected_volume[i] = white_tophat(
                    volume[i], structure=structuring_element)

        return corrected_volume
    except Exception as e:
        logger.error(f"Error during top-hat background subtraction: {e}")
        raise


def downsample_volume(array: np.ndarray, factors: Tuple[Union[int, float], ...]) -> np.ndarray:
    """
    Reduces the size of an array by downsampling along each dimension using specified factors.

    :param array: The input array to be downsampled.
    :type array: np.ndarray
    :param factors: The factors to downsample by for each dimension of the array. Each factor must be a positive number.
    :type factors: Tuple[Union[int, float], ...]
    :return: The downsampled array.
    :rtype: np.ndarray
    """

    if not isinstance(array, np.ndarray):
        raise TypeError("Input 'array' must be a numpy ndarray.")

    if not isinstance(factors, tuple) or not all(isinstance(factor, (int, float)) and factor > 0 for factor in factors):
        raise ValueError("All 'factors' must be positive numbers.")

    try:
        scales = tuple(1 / factor for factor in factors)
        return zoom(array, scales, order=1)
    except Exception as e:
        # Catches unexpected errors from the zoom function or from incorrect scale calculations.
        raise RuntimeError("An error occurred during downsampling: " + str(e))



def enhance_and_filter_volume(volume: np.ndarray, low_percentile: float = 0, high_percentile: float = 100, acclerated: bool = False) -> np.ndarray:
    """
    Enhances the contrast of a volume using specified percentiles and applies a median filter to reduce noise.
    Optionally uses GPU acceleration for the median filtering step if `accelerated` is set to True.

    :param volume: The input volume to be processed.
    :type volume: np.ndarray
    :param low_percentile: The lower percentile to use for contrast adjustment. Values below this percentile will be adjusted to the minimum intensity.
    :type low_percentile: float Default is 0.
    :param high_percentile: The higher percentile to use for contrast adjustment. Values above this percentile will be adjusted to the maximum intensity.
    :type high_percentile: float Default is 100.
    :param accelerated: If True, uses GPU acceleration to perform the median filtering. Requires CuPy to be installed.
    :type accelerated: bool, optional Default is False.
    :return: The volume after contrast enhancement and median filtering.
    :rtype: np.ndarray
    :raises ValueError: If the percentiles are out of the [0, 100] range or if high_percentile is not greater than low_percentile.
    :raises TypeError: If the input volume is not a numpy ndarray or if percentiles are not numeric.
    :raises ImportError: If `accelerated` is True but CuPy is not installed.
    """


    def apply_3d_median_filter(volume, size=3,accelerated=False):
        """
        Applies a 3D median filter to the volume.

        :param volume: The volume to apply the filter to.
        :type volume: np.ndarray
        :param size: The size of the moving window for the filter.
        :type size: int
        :return: Filtered volume.
        :rtype: np.ndarray
        """
        if accelerated:
            try:
                import cupyx.scipy.ndimage
                import cupy as cp
                return cp.asnumpy(cupyx.scipy.ndimage.median_filter(cp.array(volume), size=size))
            except ImportError:
                raise ImportError("CuPy is not installed, but is required for accelerated processing.")
        else:
            volume = median_filter(volume, size=size)

        return volume

    def auto_adjust_contrast(volume, low, high):
        """
        Automatically adjusts the contrast of the volume based on the specified lower and higher percentiles.

        :param volume: The volume to adjust the contrast for.
        :type volume: np.ndarray
        :param low: The lower percentile value for contrast adjustment.
        :type low: float
        :param high: The higher percentile value for contrast adjustment.
        :type high: float
        :return: Contrast-enhanced volume.
        :rtype: np.ndarray
        """
        v_min, v_max = np.percentile(volume, (low, high))
        return exposure.rescale_intensity(volume, in_range=(v_min, v_max))


    if not isinstance(volume, np.ndarray):
        raise TypeError("Input 'volume' must be a numpy ndarray.")

    if not (isinstance(low_percentile, (int, float)) and isinstance(high_percentile, (int, float))):
        raise TypeError("Percentiles must be int or float.")

    if not (0 <= low_percentile <= 100) or not (0 <= high_percentile <= 100):
        raise ValueError("Percentiles must be within the range [0, 100].")

    if high_percentile <= low_percentile:
        raise ValueError("high_percentile must be greater than low_percentile.")

    try:
        volume = auto_adjust_contrast(volume, low_percentile, high_percentile)
        volume = apply_3d_median_filter(volume, 5, acclerated)
    except Exception as e:
        # Catches and rethrows an error with a more user-friendly message
        raise RuntimeError(f"Failed to process the volume: {e}")
    
    return volume
