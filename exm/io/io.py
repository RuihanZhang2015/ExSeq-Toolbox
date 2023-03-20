"""
Functions to assist in folder creation and reading/writing image files. 
"""

import os
import numpy as np
import h5py
import pandas as pd
from nd2reader import ND2Reader
import statistics
from tifffile import imread
# from .image import imAdjust
from PIL import Image
import skimage.measure
from IPython.display import Image as Img2


# TODO document the expected Xlsx structure 
def readXlsx(xlsx_file):
    r"""Reads the experiment xlsx_file and returns it as a Pandas dataframe. 
    Args:
        xlsx_file (str): Path to the xlsx file. 
    """
    df = pd.read_excel(
        open(xlsx_file, 'rb'),
        engine='openpyxl',
        header = [1],
        sheet_name=3)
    
    # drop invalid rows
    flag = []
    for x in df['Point Name']:
        if isinstance(x, str) and ('#' in x):
            flag.append(False)
        else:
            flag.append(True)
    df = df.drop(df[flag].index)
    flag = []
    for x in df['X Pos[µm]']:
        if isinstance(x, float) or isinstance(x, int):
            flag.append(False)
        else:
            flag.append(True)
    df = df.drop(df[flag].index)
    
    # select columns
    zz, yy, xx = np.array(df['Z Pos[µm]'].values), np.array(df['Y Pos[µm]'].values), np.array(df['X Pos[µm]'].values)
    ii = np.array([int(x[1:])-1 for x in df['Point Name'].values])
    # need to flip x
    out = np.vstack([zz,yy,-xx,ii]).T.astype(float)
    if (ii==0).sum() != 1:
        loop_ind = np.hstack([np.where(ii==0)[0], len(ii)])
        loop_len = loop_ind[1:] - loop_ind[:-1]
        print('exist %d multipoint loops with length' % len(loop_len), loop_len)
        mid = np.argmax(loop_len)
        out = out[loop_ind[mid]:loop_ind[mid+1]]
        # take the longest one
    return out 

# TODO document the expected Xlsx structure
def readNd2(nd2_file, do_info = True):
    r"""Returns the image and metadata from the specified Nd2 file. 
    Args:
        nd2_file (str): file path.
        do_info (boolean): whether or not the Nd2 file has metadata. Default: ``True``
    """
    vol = ND2Reader(nd2_file)
    info = {}
    if do_info: 
        meta = vol.metadata
        # assume zyx order
        info['tiles_size'] = np.array([meta['z_levels'][-1]+1, meta['height'], meta['width']])
        zz = np.array(meta['z_coordinates'])
        zz_res = statistics.mode(np.round(10000 * (zz[1:]-zz[:-1])) / 10000)
        info['resolution'] = np.array([zz_res, meta['pixel_microns'], meta['pixel_microns']])
        info['channels'] = meta['channels']
    return vol, info

def tiff2H5(tiff_file, h5_file, chunk_size=(100,1024,1024), step=100, im_thres=None):
    r"""Reads the specified tiff file and re-saves it as a H5 file. 
    Args:
        tiff_file (str): path to the existing tiff file.
        h5_file (str): path to the new H5 file. 
        chunk_size (tuple): chunck size to break the image into. Default: ``(100,1024,1024)``
        step (int): z step size. Default: ``100``
        im_thresh (int, optional): integer used for image thresholding. Default: ``None``
    """
    # get tiff volume dimension
    img = Image.open(tiff_file)
    num_z = img.n_frames
    test_page = imread(tiff_file, key=range(1))
    sz = [num_z, test_page.shape[0], test_page.shape[1]]

    fid = h5py.File(h5_file, 'w')
    dtype = np.uint8 if im_thres is not None else test_page.dtype
    ds = fid.create_dataset('main', sz, compression="gzip", dtype=dtype, chunks=chunk_size)
    
    num_zi = (sz[0]+step-1) // step
    for zi in range(num_zi):
        z = min((zi+1)*step, sz[0])
        im = imread(tiff_file, key=range(zi*step, z))
        if im_thres is not None:
            im = imAdjust(im, im_thres).astype(np.uint8)
        ds[zi*step:z] = im
    fid.close()

def nd2ToVol(filename: str, fov: int, channel_name: str = '405 SD',ratio = 1):
    r"""Reads the specified Nd2 file and returns it as an array. 
    Args:
        filename (str): path of the Nd2 file. 
        fov (int): the field of view to be returned. 
        channel_name (str): the channel to be returned. 
        ratio (int): downsampling factor. Default: ``1``
    """
    # volume in zyx order
    
    vol = ND2Reader(filename)
    channel_names = vol.metadata['channels']
    # print('Available channels:', channel_names)
    channel_id = [x for x in range(len(channel_names)) if channel_name in channel_names[x]]
    assert len(channel_id) == 1
    channel_id = channel_id[0]

    out = np.zeros([len(vol)//ratio, vol[0].shape[0]//ratio , vol[0].shape[1] //ratio], np.uint16)
    for z in range(len(vol)//ratio):
        out[z] = vol.get_frame_2D(c=channel_id, t=0, z=int(z*ratio), x=0, y=0, v=fov)[::ratio,::ratio]
    return out

def nd2ToChunk(filename: str, fov: int, z_min: int, z_max :int, channel_name: str = '405 SD'):
    r"""Reads the speficied Nd2 file and returns a chunk from it. 
    Args:
        filename (str): configuration options.
        fov (int): the field of view to be returned. 
        z_min (int): starting z position of the chunk. 
        z_max (int): ending z position of the chunk. 
        channel_name (str): the channel to be returned. Default: ``'405 SD'``
    """
    # volume in zyx order
    
    vol = ND2Reader(filename)
    channel_names = vol.metadata['channels']
    # print('Available channels:', channel_names)
    channel_id = [x for x in range(len(channel_names)) if channel_name in channel_names[x]]
    assert len(channel_id) == 1
    channel_id = channel_id[0]

    out = np.zeros([z_max-z_min, vol[0].shape[0], vol[0].shape[1]], np.uint16)
    for z in range(z_max-z_min):
        out[z] = vol.get_frame_2D(c=channel_id, t=0, z=z+z_min, x=0, y=0, v=fov)
    return out

def nd2ToSlice(filename: str, fov: int, z: int, channel_name: str = '405 SD'):
    r"""Reads the speficied Nd2 file and returns a slice from it. 
    Args:
        filename (str): path of the Nd2 file. 
        fov (int): the field of view to be returned. 
        z (int): index of z slice to be returned.
        channel_name (str): the channel to be returned. Default: ``'405 SD'``
    """
    # volume in zyx order
    
    vol = ND2Reader(filename)
    channel_names = vol.metadata['channels']
    channel_id = [x for x in range(len(channel_names)) if channel_name in channel_names[x]]
    assert len(channel_id) == 1
    channel_id = channel_id[0]

    out = vol.get_frame_2D(c=channel_id, t=0, z=int(z), x=0, y=0, v=fov)
    return out


def createFolderStruc(out_dir, codes):
    r"""Creates a results folder for the specified code. 
    Args:
        outdir (str): the directory where all results for the specified code should be stored. 
        codes (list): the list of codes create the folder structure for. 
    """
    
    processed_dir = os.path.join(out_dir,"processed/")
    puncta_dir = os.path.join(out_dir,"puncta/")
    puncta_inspect_dir = os.path.join(puncta_dir,"inspect_puncta/")

    if os.path.isdir(processed_dir) is False:
        os.makedirs(processed_dir)
    
    if os.path.isdir(puncta_dir) is False:
        os.makedirs(puncta_dir)

    if os.path.isdir(puncta_inspect_dir) is False:
        os.makedirs(puncta_inspect_dir)




    for code in codes: 
        
        code_path = os.path.join(processed_dir,'code{}'.format(code))
        
        if os.path.isdir(code_path) is False:
            os.makedirs(code_path)
        
        tform_dir = os.path.join(code_path, 'tforms')
        
        if os.path.isdir(tform_dir) is False:
            os.makedirs(tform_dir)

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
    Args:
        arr (np.array): array to downsample.
        block_size (np.array): array containing down-sampling integer factor along each axis.
    """
    block_list = [block_size]*arr.ndim
    block = tuple(block_list)
    assert len(block) == arr.ndim, "block size does not match vector shape"

    new_array = skimage.measure.block_reduce(arr, block, np.mean)

    return new_array

def parseSitkLog(log_path: str):
    r"""Open the SimpleITK log and return the resulting metric and stepsize. 
    Args:
        log_path (str): path to the SimpleITK log. 
    """
    result_metric = []
    result_stepsize = []
    start_ind = 10000000
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for ind, x in enumerate(lines):
            if x == '1:ItNr\t2:Metric\t3a:Time\t3b:StepSize\t4:||Gradient||\tTime[ms]\n':
                start_ind = ind
            if ind > start_ind and '\t-' in x:
                splt = x.split('\t')
                result_metric.append(splt[1])
                result_stepsize.append(splt[3])
                
    result_metric = np.asarray(result_metric, dtype = 'float32')
    result_stepsize = np.asarray(result_stepsize, dtype = 'float32')
                
    return result_metric, result_stepsize

def saveGif(img1, img2, filename):
    r"""Takes in two images, appends one behind the other, and loops between them in a GIF. Saves and returns resulting GIF.
    Args:
        img1 (np.array): the first image to be displayed.
        img2 (np.array): the second image to be displayed.
        filename (str): the filename for saving the GIF. 
    """
    im1 =  Image.fromarray(img1)
    im2 =  Image.fromarray(img2)
    im1.save(filename, format='GIF',
                   append_images=[im2],
                   save_all=True,
                   duration=300, loop=0)
    return Img2(filename=filename)
