import os, sys
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

def readXlsx(xlsx_file):
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


def readNd2(nd2_file, do_info = True):
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
    # volume in zyx order
    
    vol = ND2Reader(filename)
    channel_names = vol.metadata['channels']
    channel_id = [x for x in range(len(channel_names)) if channel_name in channel_names[x]]
    assert len(channel_id) == 1
    channel_id = channel_id[0]

    out = vol.get_frame_2D(c=channel_id, t=0, z=int(z), x=0, y=0, v=fov)
    return out


def createFolderStruc(out_dir: str, code: str):
    
    ###################################################################
    # USAGE: place the path of where you would like results to be saved
    # in out_dir, along with date data were collected and code ########
    ###################################################################
    
    if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir)
    
    code_dir = 'code{}'.format(code)
    code_path = os.path.join(out_dir,code_dir)
    
    if os.path.isdir(code_path) is False:
        os.makedirs(code_path)
    
    tform_dir = os.path.join(code_path, 'tforms')
    
    if os.path.isdir(tform_dir) is False:
        os.makedirs(tform_dir)
    
    gif_parent_path = os.path.join(code_path, 'gifs')
    
    if os.path.isdir(gif_parent_path) is False:
        os.makedirs(gif_parent_path)
    
    gif_dirs = ['xy','zy','zx']
    
    for gif_dir in gif_dirs:
        
        gif_path = os.path.join(gif_parent_path, gif_dir)
        
        if os.path.isdir(gif_path) is False:
            os.makedirs(gif_path)
    
    print('creating paths done')

def downsample(arr, block_size):
    block_list = [block_size]*arr.ndim
    block = tuple(block_list)
    assert len(block) == arr.ndim, "block size does not match vector shape"

    new_array = skimage.measure.block_reduce(arr, block, np.mean)

    return new_array

def parseSitkLog(log_path: str):
    
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
    im1 =  Image.fromarray(img1)
    im2 =  Image.fromarray(img2)
    im1.save(filename, format='GIF',
                   append_images=[im2],
                   save_all=True,
                   duration=300, loop=0)
    return Img2(filename=filename)