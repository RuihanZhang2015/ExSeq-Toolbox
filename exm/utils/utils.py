import os
import pickle
import h5py
import numpy as np
from IPython.display import display
from PIL import Image

def chmod(path):
    r"""Sets permissions so that users and the owner can read, write and execute files at the given path. 
    Args:
        path (str): path in which privileges should be granted.
    """
    if os.name != "nt": #Skip for windows OS
        os.system('chmod 766 {}'.format(path))

def retrieve_all_puncta(args, fov):
    r"""Returns all identified puncta for a given field of view. 
    Args:
        args (args.Args): configuration options.
        fov (int): field of fiew to return. 
    """
    with open(args.work_path + '/fov{}/result.pkl'.format(fov), 'rb') as f:
        return pickle.load(f)

def retrieve_one_puncta(args, fov, puncta_index):
    r"""Returns information about a single puncta, given a specified field of view and index. 
    Args:
        args (args.Args): configuration options.
        fov (int): field of fiew.
        puncta_index (int): index of the puncta of interest. 
    """
    return retrieve_all_puncta(args,fov)[puncta_index]

def retrieve_img(args, fov, code, channel, ROI_min, ROI_max):
    r"""Returns the middle slice of a specified volume chunk. 
    Args:
        args (args.Args): configuration options.
        fov (int): the field of fiew of the volume slice to be returned. 
        code (int): the code of the volume slice to be returned. 
        channel (int): the channel of the volume slice to be returned. 
        ROI_min (list): minimum coordinates of the volume chunk to take the middle slice of. Expects coordinates in the format of [z, y, x]. 
        ROI_max (list): maximum coordinates of the volume chunk to take the middle slice of. Expects coordinates in the format of [z, y, x]. 
    """

    if ROI_min != ROI_max:
        zz = int((ROI_min[0]+ROI_max[0])//2)

    with h5py.File(args.h5_path.format(code,fov), "r") as f:
        im = f[args.channel_names[channel]][zz, max(0,ROI_min[1]):min(2048,ROI_max[1]), max(0,ROI_min[2]):min(2048,ROI_max[2])]
        im = np.squeeze(im)

    return im
    
def retrieve_vol(args, fov, code, c, ROI_min, ROI_max):
    r"""Returns a specified volume chunk. 
    Args:
        args (args.Args): configuration options.
        fov (int): the field of fiew of the volume chunk to be returned. 
        code (int): the code of the volume chunk to be returned. 
        channel (int): the channel of the volume chunk to be returned. 
        ROI_min (list): minimum coordinates of the volume chunk. Expects coordinates in the format of [z, y, x]. 
        ROI_max (list): maximum coordinates of the volume chunk. Expects coordinates in the format of [z, y, x]. 
    """
    with h5py.File(args.h5_path.format(code,fov), "r") as f:
        vol = f[args.channel_names[c]][max(0,ROI_min[0]):ROI_max[0],max(0,ROI_min[1]):min(2048,ROI_max[1]),max(0,ROI_min[2]):min(2048,ROI_max[2])]    
    return vol

# Convenient short hand for a visualization function
def display_img(img):
    if img.dtype is np.dtype(bool):
        display(Image.fromarray((img*255).astype(np.uint8)))
    else:
        display(Image.fromarray(img.astype(np.uint8)))


# TODO clean refactor utils function
# TODO clean unsed functions 
  
# def retrieve_(args,fov):
#     with open(args.args.work_path + '/fov{}/result.pkl'.format(fov), 'rb') as f:
#         return pickle.load(f)
    
# def retrieve_puncta(args,fov,puncta_index):
#     return args.retrieve_result(fov)[puncta_index]

# def retrieve_complete(args,fov):
#     with open(args.args.work_path+'/fov{}/complete.pkl'.format(fov),'rb') as f:
#         return pickle.load(f)
    
# def retrieve_coordinate(args):
#     with open(args.args.layout_file,encoding='utf-16') as f:
#         contents = f.read()

#         contents = contents.split('\n')
#         contents = [line for line in contents if line and line[0] == '#' and 'SD' not in line]
#         contents = [line.split('\t')[1:3] for line in contents]

#         coordinate = [[float(x) for x in line] for line in contents ]
#         coordinate = np.asarray(coordinate)

#         # print('oooold',coordinate[:10])

#         coordinate[:,0] = max(coordinate[:,0]) - coordinate[:,0]
#         coordinate[:,1] -= min(coordinate[:,1])
#         coordinate = np.round(np.asarray(coordinate/0.1625/(0.90*2048))).astype(int)
#         return coordinate

# def retrieve_coordinate2(args):
import xml.etree.ElementTree 
def get_offsets(filename):
    r"""Given the filename for the BDV/H5 XML file, returns the stitching offset as a (N,3) array in the ((X,Y,Z),...) order. Returned values are expressed in µm.
    Args:
        filename (str): the file name of the BDV/H5 XML file, produced by the Big Stitcher plugin of fiji. 
    """
    tree = xml.etree.ElementTree.parse(filename)
    root = tree.getroot()
    vtrans = list()
    for registration_tag in root.findall('./ViewRegistrations/ViewRegistration'):
        tot_mat = np.eye(4, 4)
        for view_transform in registration_tag.findall('ViewTransform'):
            affine_transform = view_transform.find('affine')
            mat = np.array([float(a) for a in affine_transform.text.split(" ")] + [0, 0, 0, 1]).reshape((4, 4))
            tot_mat = np.matmul(tot_mat, mat)
        vtrans.append(tot_mat)
    def transform_to_translate(m):
        m[0, :] = m[0, :] / m[0][0]
        m[1, :] = m[1, :] / m[1][1]
        m[2, :] = m[2, :] / m[2][2]
        return m[:-1, -1]

    trans = [transform_to_translate(vt).astype(np.int64) for vt in vtrans]
    return np.stack(trans)

# coordinate = get_offsets()
# coordinate = np.asarray(coordinate)

# coordinate[:,0] = max(coordinate[:,0]) - coordinate[:,0]
# coordinate[:,1] -= min(coordinate[:,1])
# coordinate[:,2] -= min(coordinate[:,2])
# # coordinate[:,:2] = np.asarray(coordinate[:,:2]/0.1625)
# # coordinate[:,2] = np.asarray(coordinate[:,2]/0.4)
# return coordinate

# ### Visualization
# doc = Document(sheet_path)
# sheets = doc.sheets()
# tables = sheets[0].tables()
# data = tables[0].rows(values_only=True)

# df = pd.DataFrame(data[1:], columns=data[0])


# args.map_gene = collections.defaultdict(list)
# for i in range(len(df)):
#     temp = df.loc[i,'Barcode']
#     temp = ''.join([args.code2num[temp[code]] for code in args.codes])
#     args.map_gene[temp] = df.loc[i,'Gene']

# colors = sns.color_palette(None, len(args.map_gene.keys()))
# args.map_color = {a:b for a,b in zip(args.map_gene.keys(),colors)}

# if not gene_list and not hasattr(args,'gene_list'):
#     args.gene_list = 'gene_list.numbers'

# if not layout_file and not hasattr(args,'layout_file'):
#     args.layout_file = project_path + 'code0/out.csv'
