"""
Functions to assist in folder creation and reading/writing image files. 
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

from exm.utils.log import configure_logger
logger = configure_logger('ExSeq-Toolbox')


# TODO document the expected Xlsx structure
def readXlsx(xlsx_file):
    r"""Reads the experiment xlsx_file and returns it as a Pandas dataframe.

    :param str xlsx_file: Path to the ``xlsx`` file.
    """
    df = pd.read_excel(
        open(xlsx_file, "rb"), engine="openpyxl", header=[1], sheet_name=3
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
        logger.info("exist %d multipoint loops with length" % len(loop_len), loop_len)
        mid = np.argmax(loop_len)
        out = out[loop_ind[mid] : loop_ind[mid + 1]]
        # take the longest one
    return out


# TODO document the expected Xlsx structure
def readNd2(nd2_file, do_info=True):
    r"""Returns the image and metadata from the specified Nd2 file.

    :param str nd2_file: file path.
    :param bool do_info: whether or not the Nd2 file has metadata. Default: ``True``
    """
    vol = ND2Reader(nd2_file)
    info = {}
    if do_info:
        meta = vol.metadata
        # assume zyx order
        info["tiles_size"] = np.array(
            [meta["z_levels"][-1] + 1, meta["height"], meta["width"]]
        )
        zz = np.array(meta["z_coordinates"])
        zz_res = statistics.mode(np.round(10000 * (zz[1:] - zz[:-1])) / 10000)
        info["resolution"] = np.array(
            [zz_res, meta["pixel_microns"], meta["pixel_microns"]]
        )
        info["channels"] = meta["channels"]
    return vol, info


def tiff2H5(tiff_file, h5_file, chunk_size=(100, 1024, 1024), step=100, im_thres=None):
    r"""Reads the specified tiff file and re-saves it as a H5 file.

    :param str tiff_file: path to the existing ``tiff`` file.
    :param str h5_file: path to the new ``H5`` file.
    :param tuple chunk_size: chunk size to break the image into. Default: :math:`(100, 1024, 1024)`
    :param int step: :math:`z` step size. Default: :math:`100`
    :param im_thresh: integer used for image thresholding. Default: ``None``
    :type im_thresh: int, optional
    """
    # get tiff volume dimension
    img = Image.open(tiff_file)
    num_z = img.n_frames
    test_page = imread(tiff_file, key=range(1))
    sz = [num_z, test_page.shape[0], test_page.shape[1]]

    fid = h5py.File(h5_file, "w")
    dtype = np.uint8 if im_thres is not None else test_page.dtype
    ds = fid.create_dataset(
        "main", sz, compression="gzip", dtype=dtype, chunks=chunk_size
    )

    num_zi = (sz[0] + step - 1) // step
    for zi in range(num_zi):
        z = min((zi + 1) * step, sz[0])
        im = imread(tiff_file, key=range(zi * step, z))
        if im_thres is not None:
            im = imAdjust(im, im_thres).astype(np.uint8)
        ds[zi * step : z] = im
    fid.close()


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
            raise ValueError(f"Invalid channel name: {channel_name}. Please provide a valid channel name.")
        
        channel_id = channel_id[0]

        out = np.zeros(
            [len(vol) // ratio, vol[0].shape[0] // ratio, vol[0].shape[1] // ratio],
            np.uint16,
        )
        for z in range(len(vol) // ratio):
            out[z] = vol.get_frame_2D(c=channel_id, t=0, z=int(z * ratio), x=0, y=0, v=fov)[
                ::ratio, ::ratio
            ]
        return out

    except Exception as e:
        print(f"Error occurred while converting ND2 file to volume: {e}")
        raise


def nd2ToChunk(
    filename: str, fov: int, z_min: int, z_max: int, channel_name: str = "405 SD"
):
    r"""Reads the speficied Nd2 file and returns a chunk from it.

    :param str filename: configuration options
    :param int fov: the field of view to be returned.
    :param int z_min: starting :math:`z` position of the chunk.
    :param int z_max: ending :math:`z` position of the chunk.
    :param str channel_name: the channel to be returned. Default: ``'405 SD'``
    """
    # volume in zyx order

    vol = ND2Reader(filename)
    channel_names = vol.metadata["channels"]
    channel_id = [
        x for x in range(len(channel_names)) if channel_name in channel_names[x]
    ]
    assert len(channel_id) == 1
    channel_id = channel_id[0]

    out = np.zeros([z_max - z_min, vol[0].shape[0], vol[0].shape[1]], np.uint16)
    for z in range(z_max - z_min):
        out[z] = vol.get_frame_2D(c=channel_id, t=0, z=z + z_min, x=0, y=0, v=fov)
    return out


def nd2ToSlice(filename: str, fov: int, z: int, channel_name: str = "405 SD"):
    r"""Reads the speficied Nd2 file and returns a slice from it.

    :param str filename: path of the ``ND2`` file.
    :param int fov: the field of view to be returned.
    :param int z: index of :math:`z` slice to be returned.
    :param str channel_name: the channel to be returned. Default: ``'405 SD'``
    """
    # volume in zyx order

    vol = ND2Reader(filename)
    channel_names = vol.metadata["channels"]
    channel_id = [
        x for x in range(len(channel_names)) if channel_name in channel_names[x]
    ]
    assert len(channel_id) == 1
    channel_id = channel_id[0]

    out = vol.get_frame_2D(c=channel_id, t=0, z=int(z), x=0, y=0, v=fov)
    return out


def create_folder_structure(processed_dir: str, fovs:List[int], codes: List[int]) -> None:
    r"""
    Creates a results folder for the specified codes.

    :param processed_dir: The directory where all results for the specified codes should be stored.
    :type processed_dir: str
    :param fovs: The list of Fovs to create the folder structure for.
    :type fovs: List[int]
    :param codes: The list of codes to create the folder structure for.
    :type codes: List[int]
    """
    try:
        processed_dir = Path(processed_dir)
        puncta_dir = processed_dir.joinpath("puncta/")
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


def downsample(arr, block_size):
    r"""Takes in a single or multidimensional array and downsampled it using skimage.measure.block_reduce.

    :param numpy.array arr: array to downsample.
    :param numpy.array block_size: array containing down-sampling integer factor along each axis.
    """
    block_list = [block_size] * arr.ndim
    block = tuple(block_list)
    assert len(block) == arr.ndim, "block size does not match vector shape"

    new_array = skimage.measure.block_reduce(arr, block, np.mean)

    return new_array


def parseSitkLog(log_path: str):
    r"""Open the SimpleITK log and return the resulting metric and stepsize.

    :param str log_path: path to the SimpleITK log.
    """
    result_metric = []
    result_stepsize = []
    start_ind = 10000000
    with open(log_path, "r") as f:
        lines = f.readlines()
        for ind, x in enumerate(lines):
            if (
                x
                == "1:ItNr\t2:Metric\t3a:Time\t3b:StepSize\t4:||Gradient||\tTime[ms]\n"
            ):
                start_ind = ind
            if ind > start_ind and "\t-" in x:
                splt = x.split("\t")
                result_metric.append(splt[1])
                result_stepsize.append(splt[3])

    result_metric = np.asarray(result_metric, dtype="float32")
    result_stepsize = np.asarray(result_stepsize, dtype="float32")

    return result_metric, result_stepsize


def saveGif(img1, img2, filename):
    r"""Takes in two images, appends one behind the other, and loops between them in a GIF. Saves and returns resulting GIF.

    :param numpy.array img1: the first image to be displayed.
    :param numpy.array img2: the second image to be displayed.
    :param str filename: the filename for saving the GIF.
    """
    im1 = Image.fromarray(img1)
    im2 = Image.fromarray(img2)
    im1.save(
        filename, format="GIF", append_images=[im2], save_all=True, duration=300, loop=0
    )
    return Img2(filename=filename)
