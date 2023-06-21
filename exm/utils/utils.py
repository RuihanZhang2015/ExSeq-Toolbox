import os
import pickle
import h5py
import random
import pandas as pd
import numpy as np
from IPython.display import display
from PIL import Image

from exm.utils import configure_logger
logger = configure_logger('ExSeq-Toolbox')


def chmod(path):
    r"""Sets permissions so that users and the owner can read, write and execute files at the given path.

    :param str path: path in which privileges should be granted
    """
    if os.name != "nt":  # Skip for windows OS
        os.system("chmod 766 {}".format(path))


def retrieve_all_puncta(args, fov):
    r"""Returns all identified puncta for a given field of view.

    :param args.Args args: configuration options.
    :param int fov: field of view to return
    """
    with open(args.work_path + "/fov{}/result.pkl".format(fov), "rb") as f:
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
            max(0, ROI_min[1]) : min(2048, ROI_max[1]),
            max(0, ROI_min[2]) : min(2048, ROI_max[2]),
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
            max(0, ROI_min[0]) : ROI_max[0],
            max(0, ROI_min[1]) : min(2048, ROI_max[1]),
            max(0, ROI_min[2]) : min(2048, ROI_max[2]),
        ]
    return vol

def gene_barcode_mapping(args):
    r"""This function loads a CSV file `args.gene_digit_csv` containing gene symbols and corresponding barcodes. It converts the barcodes to digit representations and creates two mappings: 'digit2gene' (from digit representation to gene symbol) and 'gene2digit' (from gene symbol to digit representation). These mappings are useful for identifying genes associated with puncta barcode in a field of view.
    :param args.Args args: configuration options, including the path to the gene-digit CSV file and the mapping from code to number.
    :returns: A tuple containing three elements. The first element is a pandas DataFrame containing the original CSV data with an additional column for the digit representations of the barcodes. The second element is a dictionary mapping from digit representation to gene symbol ('digit2gene'). The third element is a dictionary mapping from gene symbol to digit representation ('gene2digit').
    """
    df = pd.read_csv(args.gene_digit_csv)
    df['Digits'] = [''.join([args.code2num[c] for c in barcode]) for barcode in df['Barcode']]
    digit2gene,gene2digit = {},{}
    for i, row in df.iterrows():
        digit2gene[row['Digits']] = row['Symbol']
        gene2digit[row['Symbol']] = row['Digits']
    return df,digit2gene,gene2digit


def display_img(img):
    r"""
    This function displays an image using the Image module from the Python Imaging Library (PIL). The function supports images of type boolean and other numpy data types. For boolean images, the function multiplies the image by 255 to create an 8-bit grayscale image. For non-boolean images, the function simply converts the image to an 8-bit grayscale image.
    :param numpy.ndarray img: The input image to display. This can be a boolean or non-boolean numpy array.
    """
    if img.dtype is np.dtype(bool):
        display(Image.fromarray((img * 255).astype(np.uint8)))
    else:
        display(Image.fromarray(img.astype(np.uint8)))


def retrieve_digit(args, digit):
    """This function retrieves all puncta with a specified barcode (represented as a digit) across all provided fields of view (fov). For each puncta that matches the barcode, it appends the puncta (with added fov information) to a list.
    :param args.Args args:  Configuration options, including the list of fovs and the method to retrieve all puncta for a given fov.
    :param digit: The barcode to search for, represented as a digit.
    :returns: A list of all puncta across all fovs that match the specified barcode. Each puncta is represented as a dictionary containing puncta information and the fov it was found in.
    """
    puncta_lists = []
    for fov in args.fovs:
        result = retrieve_all_puncta(args,fov)
        for puncta in result:
            if puncta['barcode'] == digit:
                puncta_lists.append({
                    **puncta,
                    'fov':fov
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
        result = retrieve_all_puncta(args,fov)
        for entry in result:
            summary['number'][entry['barcode']] += 1
            summary[f'fov{fov}'][entry['barcode']] += 1
    summary = pd.DataFrame(summary).fillna(0).astype(int)
    summary = summary.sort_values(by='number', ascending=False)
    summary.to_csv(os.path.join(args.work_path,'digit_summary.csv'))
    return summary

def retrieve_complete(args):
    r"""This function retrieves a complete summary of barcodes that are present in both the gene-barcode mapping and the overall barcode summary. The function returns a DataFrame sorted by gene names and also writes this DataFrame to a CSV file.
    :param args.Args args: configuration options, including methods for gene-barcode mapping, retrieving barcode summary, and the work path where the summary CSV file will be saved.
    :returns: A pandas DataFrame containing the complete summary of barcodes. The DataFrame is indexed by barcode with columns for total count ('number') and count per fov (e.g., 'fov1', 'fov2', ...). Additionally, it contains a 'gene' column that maps each barcode to its corresponding gene. The DataFrame is sorted by gene names in ascending order.

    """
    df,digit2gene,gene2digit = gene_barcode_mapping(args)
    summary = retrieve_summary(args)
    complete = summary.loc[list(set(df['Digits']) & set(summary.index))]
    complete['gene'] = [digit2gene[digit] for digit in complete.index]
    complete = complete.sort_values('gene') 
    complete.to_csv(os.path.join(args.work_path,'gene_summary.csv'))

    return complete

def retrieve_gene(args, gene):
    r"""This function retrieves all puncta associated with a specific gene across all fields of view (fovs). It leverages a Hamming distance function to match the barcode of each puncta with the gene of interest, permitting a maximum of one mismatch. It also writes the gene-barcode mapping to a CSV file.
    :param args.Args args: configuration options, including methods for gene-barcode mapping, retrieving all puncta, and the work path where the gene-barcode CSV file will be saved.
    :param str gene: The gene of interest for which all corresponding puncta across all fovs will be retrieved.
    :returns: A list of dictionaries, each representing a puncta associated with the gene. Each dictionary includes the puncta's properties, as well as the fov in which it is found.
    """
    def within_hamming_distance(a,b):
        diff = 0
        for x,y in zip(a,b):
            if x!=y:
                diff +=1
            if diff>=2:
                return False
        return True

    df,digit2gene,gene2digit = gene_barcode_mapping(args)
    digit = gene2digit[gene]

    puncta_lists = []
    for fov in args.fovs:
        result = retrieve_all_puncta(args,fov)
        for puncta in result:
            if within_hamming_distance(puncta['barcode'],digit):
                puncta_lists.append({
                    **puncta,
                    'fov':fov
                })
    df.to_csv(os.path.join(args.work_path,'gene_{}_digit_map.csv'.format(gene)))
    return puncta_lists

def generate_debug_candidate(args, gene = None, fov = None, num_missing_code = 1):
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
        logger.info('Studying gene {} in fov '.format(gene),fov)
        puncta_lists = [puncta for puncta in puncta_lists if puncta['fov'] == fov]
    else:
        logger.info('Studying gene {} in all fovs'.format(gene))
    logger.info('Total barcode that matches gene {} '.format(gene),len(puncta_lists))

    puncta_lists = [puncta for puncta in puncta_lists if puncta['barcode'].count('_') == num_missing_code]
    if len(puncta_lists) == 0:
        logger.info('Total barcode with {} missing codes: {}'.format(num_missing_code,0))
        return generate_debug_candidate(args, gene = None, fov = None, num_missing_code = 1)
    else:
        logger.info('Total barcode with {} missing codes: {}'.format(num_missing_code,len(puncta_lists)))

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
                [float(a) for a in affine_transform.text.split(" ")] + [0, 0, 0, 1]
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
