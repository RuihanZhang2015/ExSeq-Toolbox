"""
The IO module in the ExSeq Toolbox is designed to streamline the process of reading, converting, and managing image data files for expansion microscopy
"""

import os
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from nd2reader import ND2Reader
import statistics
from tifffile import imread
from typing import List

# from .image import imAdjust
from PIL import Image
import skimage.measure
from IPython.display import Image as Img2

from typing import Optional, Dict, List, Tuple, Any

from exm.utils.log import configure_logger
logger = configure_logger('ExSeq-Toolbox')


# TODO document the expected Xlsx structure
def readXlsx(xlsx_file: str) -> np.ndarray:
    r"""
    Reads the experiment xlsx_file and returns it as a NumPy array.

    The function expects the following columns to be present: 'Point Name', 'Z Pos[µm]', 'Y Pos[µm]', and 'X Pos[µm]'.
    It processes the file to extract position data and returns this data, transforming 'X Pos' by negating its values.

    :param xlsx_file: Path to the xlsx file.
    :type xlsx_file: str
    :return: A NumPy array containing the processed position data.
    :rtype: np.ndarray

    :raises FileNotFoundError: If the xlsx file cannot be found.
    :raises ValueError: If the file structure is not as expected.
    """
    try:
        df = pd.read_excel(
            xlsx_file, engine="openpyxl", header=[1], sheet_name=3
        )

        # drop invalid rows
        flag = []
        for x in df["Point Name"]:
            if isinstance(x, str) and ("#" in x):
                flag.append(False)
            else:
                flag.append(True)
        df = df.drop(df[flag].index)
        flag = []
        for x in df["X Pos[µm]"]:
            if isinstance(x, float) or isinstance(x, int):
                flag.append(False)
            else:
                flag.append(True)
        df = df.drop(df[flag].index)

        # select columns
        zz, yy, xx = (
            np.array(df["Z Pos[µm]"].values),
            np.array(df["Y Pos[µm]"].values),
            np.array(df["X Pos[µm]"].values),
        )
        ii = np.array([int(x[1:]) - 1 for x in df["Point Name"].values])
        # need to flip x
        out = np.vstack([zz, yy, -xx, ii]).T.astype(float)
        if (ii == 0).sum() != 1:
            loop_ind = np.hstack([np.where(ii == 0)[0], len(ii)])
            loop_len = loop_ind[1:] - loop_ind[:-1]
            logger.info("exist %d multipoint loops with length" %
                        len(loop_len), loop_len)
            mid = np.argmax(loop_len)
            out = out[loop_ind[mid]: loop_ind[mid + 1]]
            # take the longest one
        return out
    except FileNotFoundError as e:
        logger.error(f"Xlsx file not found: {e}")
        raise FileNotFoundError(f"Xlsx file not found: {e}")
    except ValueError as e:
        logger.error(f"Error reading xlsx file: {e}")
        raise ValueError(f"Error reading xlsx file: {e}")

# TODO document the expected Xlsx structure


def readNd2(nd2_file: str, do_info: bool = True) -> Tuple[ND2Reader, Optional[Dict[str, Any]]]:
    r"""
    Returns the image and metadata from the specified Nd2 file.

    :param nd2_file: File path to the Nd2 file.
    :type nd2_file: str
    :param do_info: Whether or not to extract metadata from the Nd2 file. Defaults to True.
    :type do_info: bool
    :return: A tuple containing ND2Reader object for the image data and a dictionary for metadata if requested.
    :rtype: Tuple[ND2Reader, Optional[Dict[str, Any]]]

    :raises FileNotFoundError: If the Nd2 file cannot be found.
    :raises Exception: If an error occurs while reading the Nd2 file or extracting metadata.
    """
    try:
        vol = ND2Reader(nd2_file)
        info: Optional[Dict[str, Any]] = None

        if do_info:
            meta = vol.metadata
            # assume zyx order
            info["tiles_size"] = np.array(
                [meta["z_levels"][-1] + 1, meta["height"], meta["width"]]
            )
            zz = np.array(meta["z_coordinates"])
            zz_res = statistics.mode(
                np.round(10000 * (zz[1:] - zz[:-1])) / 10000)
            info["resolution"] = np.array(
                [zz_res, meta["pixel_microns"], meta["pixel_microns"]]
            )
            info["channels"] = meta["channels"]
        return vol, info
    except FileNotFoundError as e:
        logger.error(f"Nd2 file not found: {e}")
        raise FileNotFoundError(f"Nd2 file not found: {e}")
    except Exception as e:
        logger.error(f"Error reading Nd2 file: {e}")
        raise Exception(f"Error reading Nd2 file: {e}")


def tiff2H5(tiff_file: str, h5_file: str, chunk_size: Tuple[int, int, int] = (100, 1024, 1024), step: int = 100, im_thres: Optional[int] = None) -> None:
    r"""
    Reads a TIFF file and re-saves it as an H5 file.

    :param tiff_file: Path to the existing TIFF file.
    :type tiff_file: str
    :param h5_file: Path to the new H5 file.
    :type h5_file: str
    :param chunk_size: Chunk size to break the image into. Default: (100, 1024, 1024)
    :type chunk_size: Tuple[int, int, int]
    :param step: Z step size. Default: 100
    :type step: int
    :param im_thres: Integer used for image thresholding, None to disable. Default: None
    :type im_thres: Optional[int]

    :raises FileNotFoundError: If the TIFF file is not found.
    :raises IOError: If there is an error reading the TIFF file or writing the H5 file.
    :raises ValueError: If there are issues with the image thresholding.
    """
    try:
        # Get TIFF volume dimensions
        with Image.open(tiff_file) as img:
            num_z = img.n_frames

        test_page = imread(tiff_file, key=range(1))
        sz = [num_z, test_page.shape[0], test_page.shape[1]]

        # Open or create the H5 file
        with h5py.File(h5_file, "w") as fid:
            dtype = np.uint8 if im_thres is not None else test_page.dtype
            ds = fid.create_dataset(
                "main", sz, compression="gzip", dtype=dtype, chunks=chunk_size)

            num_zi = (sz[0] + step - 1) // step
            for zi in range(num_zi):
                z = min((zi + 1) * step, sz[0])
                im = imread(tiff_file, key=range(zi * step, z))
                if im_thres is not None:
                    im = imAdjust(im, im_thres).astype(np.uint8)
                ds[zi * step: z] = im
    except FileNotFoundError as e:
        logger.error(f"The TIFF file was not found: {e}")
        raise
    except IOError as e:
        logger.error(
            f"There was an error reading the TIFF file or writing the H5 file: {e}")
        raise
    except ValueError as e:
        logger.error(f"There was an issue with the image thresholding: {e}")
        raise


def nd2ToVol(filename: str, fov: int, channel_name: str = "405 SD", ratio: int = 1) -> np.ndarray:
    r"""
    Reads the specified Nd2 file and returns it as a numpy array.

    :param filename: Path of the ND2 file.
    :type filename: str
    :param fov: The field of view to be returned.
    :type fov: int
    :param channel_name: The channel to be returned. Default is "405 SD".
    :type channel_name: str
    :param ratio: Downsampling factor. Default is 1.
    :type ratio: int
    :return: The ND2 file converted into a numpy array.
    :rtype: np.ndarray
    """
    try:
        # volume in zyx order
        vol = ND2Reader(filename)
        channel_names = vol.metadata["channels"]

        channel_id = [
            x for x in range(len(channel_names)) if channel_name in channel_names[x]
        ]

        if len(channel_id) != 1:
            raise ValueError(
                f"Invalid channel name: {channel_name}. Please provide a valid channel name.")

        channel_id = channel_id[0]

        out = np.zeros(
            [len(vol) // ratio, vol[0].shape[0] //
                ratio, vol[0].shape[1] // ratio],
            np.uint16,
        )
        for z in range(len(vol) // ratio):
            out[z] = vol.get_frame_2D(c=channel_id, t=0, z=int(z * ratio), x=0, y=0, v=fov)[
                ::ratio, ::ratio
            ]
        return out

    except Exception as e:
        logger.error(
            f"Error occurred while converting ND2 file to volume: {e}")
        raise


def imsToVol(filename: str, volume_index: int) -> np.ndarray:
    """
    Reads a specific volume from an IMS file based on the given volume index.

    :param filename: Path of the IMS file.
    :type filename: str
    :param volume_index: Index of the volume to retrieve, which corresponds to a specific channel in the dataset.
    :type volume_index: int
    :return: The extracted volume as a NumPy array.
    :rtype: np.ndarray
    :raises ValueError: If `volume_index` is negative.
    :raises KeyError: If the specified path within the file does not exist.
    :raises FileNotFoundError: If the specified file does not exist.
    :raises OSError: If there is an issue opening the file, such as permissions or corruption.
    """
    if volume_index < 0:
        logger.error("Volume index cannot be negative.")
        raise

    try:
        with h5py.File(filename, 'r') as file:
            base_path = f'/DataSet/ResolutionLevel 0/TimePoint 0/Channel {volume_index}/Data'
            if base_path not in file:
                logger.error(f"The specified path '{base_path}' does not exist in the file.")
                raise
            volume = file[base_path][:]
            return volume
    except FileNotFoundError:
        logger.error(f"The file '{filename}' was not found.")
        raise
    except OSError as e:
        logger.error(f"An error occurred while accessing the file '{filename}': {e}")
        raise


# def get_raw_volume(volume_path: str, channel_name: str, volume_index: int) -> np.ndarray:
#     """
#     Retrieves a volume from a specified file, handling different types based on the file extension.

#     :param volume_path: Path to the volume file.
#     :type volume_path: str
#     :param channel_name: The channel name to use when extracting data from ND2 files.
#     :type channel_name: str
#     :param volume_index: The index of the volume or channel to retrieve, used for IMS files.
#     :type volume_index: int
#     :return: The extracted volume as a NumPy array.
#     :rtype: np.ndarray
#     :raises FileNotFoundError: If the file specified does not exist.
#     :raises ValueError: If the file type is unsupported.
#     """
#     # Check file existence
#     if not os.path.exists(volume_path):
#         logger.error(f"The file '{volume_path}' does not exist.")
#         raise 

#     file_type = os.path.splitext(volume_path)[1]

#     try:
#         if file_type == '.nd2':
#             volume = nd2ToVol(volume_path, 0, channel_name)  # Assumed modification needed for the index
#         elif file_type == '.ims':
#             volume = imsToVol(volume_path, volume_index)
#         else:
#             logger.error(f"Unsupported file type '{file_type}' for volume path '{volume_path}'.")
#             raise
#     except Exception as e:
#         logger.error(f"Error processing the file '{volume_path}': {e}")
#         raise

#     return volume


def nd2ToChunk(filename: str, fov: int, z_min: int, z_max: int, channel_name: str = "405 SD") -> np.ndarray:
    r"""
    Reads the specified Nd2 file and returns a chunk from it.

    :param filename: Path to the ND2 file.
    :type filename: str
    :param fov: The field of view to be returned.
    :type fov: int
    :param z_min: Starting z position of the chunk.
    :type z_min: int
    :param z_max: Ending z position of the chunk.
    :type z_max: int
    :param channel_name: The channel to be returned. Default is "405 SD".
    :type channel_name: str

    :return: A 3D numpy array representing the selected chunk of the ND2 file.
    :rtype: np.ndarray

    :raises ValueError: If the specified channel is not found or if there are multiple matches for the channel name.
    """
    try:
        vol = ND2Reader(filename)
        channel_names = vol.metadata["channels"]
        channel_id = [x for x in range(
            len(channel_names)) if channel_name in channel_names[x]]

        if len(channel_id) != 1:
            raise ValueError(
                f"Channel name {channel_name} is ambiguous or not found.")
        channel_id = channel_id[0]

        out = np.zeros([z_max - z_min, vol.sizes['y'],
                       vol.sizes['x']], np.uint16)
        for z in range(z_min, z_max):
            out[z - z_min] = vol.get_frame_2D(c=channel_id,
                                              t=0, z=z, x=0, y=0, v=fov)

        return out

    except Exception as e:
        logger.error(f"An error occurred while reading the ND2 file: {e}")
        raise


def nd2ToSlice(filename: str, fov: int, z: int, channel_name: str = "405 SD") -> np.ndarray:
    r"""
    Reads the specified Nd2 file and returns a single slice from it.

    :param filename: Path to the ND2 file.
    :type filename: str
    :param fov: The field of view to be returned.
    :type fov: int
    :param z: Index of the z slice to be returned.
    :type z: int
    :param channel_name: The channel to be returned. Default is "405 SD".
    :type channel_name: str

    :return: A 2D numpy array representing the selected slice of the ND2 file.
    :rtype: np.ndarray

    :raises ValueError: If the specified channel is not found or if there are multiple matches for the channel name.
    """
    try:
        vol = ND2Reader(filename)
        channel_names = vol.metadata["channels"]
        channel_id = [x for x in range(
            len(channel_names)) if channel_name in channel_names[x]]

        if len(channel_id) != 1:
            raise ValueError(
                f"Channel name {channel_name} is ambiguous or not found.")
        channel_id = channel_id[0]

        out = vol.get_frame_2D(c=channel_id, t=0, z=z, x=0, y=0, v=fov)
        return out

    except Exception as e:
        logger.error(f"An error occurred while reading the ND2 file: {e}")
        raise


def create_folder_structure(processed_dir: str,puncta_dir_name:str, fovs: List[int], codes: List[int]) -> None:
    r"""
    Creates a results folder for the specified codes.

    :param processed_dir: The directory where all results for the specified codes should be stored.
    :type processed_dir: str
    :param puncta_dir_name: The directory where all results for puncta analysis.
    :type processed_dir: str
    :param fovs: The list of Fovs to create the folder structure for.
    :type fovs: List[int]
    :param codes: The list of codes to create the folder structure for.
    :type codes: List[int]
    """
    try:
        processed_dir = Path(processed_dir)
        puncta_dir = processed_dir.joinpath(puncta_dir_name)
        puncta_inspect_dir = puncta_dir.joinpath("inspect_puncta/")

        processed_dir.mkdir(parents=True, exist_ok=True)
        puncta_dir.mkdir(parents=True, exist_ok=True)
        puncta_inspect_dir.mkdir(parents=True, exist_ok=True)

        for code in codes:
            code_path = processed_dir / f"code{code}"
            code_path.mkdir(exist_ok=True)

            tform_dir = code_path.joinpath("tforms")
            tform_dir.mkdir(exist_ok=True)

        align_eval_dir = processed_dir / "alignment_evaluation"
        align_eval_dir.mkdir(parents=True, exist_ok=True)

        for fov in fovs:
            fov_dir = align_eval_dir / f"FOV{fov}"
            fov_dir.mkdir(exist_ok=True)

    except Exception as e:
        print(f"Error occurred while creating folder structure: {e}")
        raise

        # TODO Do we need the gifs Dir
        # gif_parent_path = os.path.join(code_path, 'gifs')

        # if os.path.isdir(gif_parent_path) is False:
        #     os.makedirs(gif_parent_path)

        # gif_dirs = ['xy','zy','zx']

        # for gif_dir in gif_dirs:

        #     gif_path = os.path.join(gif_parent_path, gif_dir)

        #     if os.path.isdir(gif_path) is False:
        #         os.makedirs(gif_path)


def downsample(arr: np.ndarray, block_size: int) -> np.ndarray:
    """
    Downsamples a single or multidimensional array using skimage.measure.block_reduce.

    :param arr: Array to downsample.
    :type arr: np.ndarray
    :param block_size: Integer factor for down-sampling along each axis.
    :type block_size: int

    :return: Downsampled array.
    :rtype: np.ndarray

    :raises ValueError: If block size does not match the array's dimensions.
    """
    block_list = [block_size] * arr.ndim
    block = tuple(block_list)

    if len(block) != arr.ndim:
        raise ValueError("block size does not match array's dimensions")

    new_array = block_reduce(arr, block, np.mean)

    return new_array


def parse_sitk_log(log_path: str) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Parses a SimpleITK log file and returns the resulting metric and step size.

    :param log_path: Path to the SimpleITK log.
    :type log_path: str

    :return: A tuple containing arrays of metrics and step sizes.
    :rtype: Tuple[np.ndarray, np.ndarray]

    :raises FileNotFoundError: If the log file does not exist or cannot be opened.
    """
    result_metric = []
    result_stepsize = []
    start_ind = float('inf')

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            for ind, x in enumerate(lines):
                if x.startswith("1:ItNr"):
                    start_ind = ind
                if ind > start_ind and "\t-" in x:
                    splt = x.split("\t")
                    result_metric.append(splt[1])
                    result_stepsize.append(splt[3])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Log file at {log_path} could not be found or opened.")

    result_metric = np.asarray(result_metric, dtype="float32")
    result_stepsize = np.asarray(result_stepsize, dtype="float32")

    return result_metric, result_stepsize


def save_gif(img1: np.ndarray, img2: np.ndarray, filename: str) -> Image:
    r"""
    Creates a GIF by appending one image behind the other and loops between them.

    :param img1: The first image to be displayed.
    :type img1: np.ndarray
    :param img2: The second image to be displayed.
    :type img2: np.ndarray
    :param filename: The filename for saving the GIF.
    :type filename: str

    :return: The resulting GIF image.
    :rtype: PIL.Image

    :raises IOError: If there is an error saving the GIF.
    """
    try:
        im1 = Image.fromarray(img1)
        im2 = Image.fromarray(img2)
        im1.save(
            filename, format="GIF", append_images=[im2], save_all=True, duration=300, loop=0
        )
        return Image.open(filename)
    except IOError:
        raise IOError(f"Error saving GIF to {filename}")


