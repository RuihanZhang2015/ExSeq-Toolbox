"""
Stitching module provides several functions to manage and process expansion microscopy image data, particularly for blending and stitching together image tiles, handling blob deduplication, and creating files compatible with the BigDataViewer (BDV) format.
"""
import os
import xml.etree.ElementTree
from collections import defaultdict

import tifffile
import npy2bdv
import numpy as np
from nd2reader import ND2Reader
from scipy.ndimage import binary_dilation as dilate

from typing import List, Tuple, Optional, Dict, Set

from exm.utils import configure_logger
logger = configure_logger('ExSeq-Toolbox')

def get_offsets(filename: str) -> np.ndarray:
    r"""
    Extracts the transformation offsets from a BigDataViewer (BDV) or Hierarchical Data Format 5 (H5) XML file. 
    The offsets correspond to the affine transformations applied to the views in the dataset.

    Each view's transformation is represented as a 4x4 matrix, which is accumulated (multiplied) if multiple 
    transformations are present. The function returns these as an array of matrices, one for each view.

    :param filename: Path to the XML file containing view registration and transformation or raw postioning data.
    :type filename: str
    
    :return: An array of  corresponding to the offsets of each view as (N,3) array in the ((X,Y,Z),...) order.
    :rtype: np.ndarray
    """
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        tree = xml.etree.ElementTree.parse(filename)
        root = tree.getroot()
    except xml.etree.ElementTree.ParseError as e:
        logger.error(f"Error parsing XML file: {e}")
        raise
    try:
        vtrans = list()
        for registration_tag in root.findall('./ViewRegistrations/ViewRegistration'):
            tot_mat = np.eye(4, 4)
            for view_transform in registration_tag.findall('ViewTransform'):
                try:
                    affine_transform = view_transform.find('affine')
                    if affine_transform is not None:
                        mat_values = [float(a) for a in affine_transform.text.split(" ") if a != ""]
                        mat = np.array(mat_values + [0, 0, 0, 1]).reshape((4, 4))
                        tot_mat = np.matmul(tot_mat, mat)
                except ValueError as e:
                    logger.error(f"Invalid affine transform data: {e}")
                    raise

        def transform_to_translate(m):
            m[0, :] = m[0, :] / m[0][0]
            m[1, :] = m[1, :] / m[1][1]
            m[2, :] = m[2, :] / m[2][2]
            return m[:-1, -1]

        trans = [transform_to_translate(vt).astype(np.int64) for vt in vtrans]
        return np.stack(trans)
    
    except Exception as e:
        logger.error(f"An error occurred while extracting transformations from '{filename}': {e}")



def get_offsets_nd2(filename: str) -> np.ndarray:
    r"""
    Extracts the transformation offsets from a Nikon Digital Sight 2 (ND2) file. 
    The offsets correspond to the XYZ positions of each acquisition FOV.

    The function reads metadata from the ND2 file to extract the positions and 
    returns these as an array of (X, Y, Z) coordinates.

    :param filename: Path to the ND2 file containing acquisition point data.
    :type filename: str
    
    :return: An array of (X, Y, Z) coordinates for each acquisition point.
    :rtype: np.ndarray
    """
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        images = ND2Reader(filename)
        meta = images._parser._raw_metadata.image_metadata
        raw_translations = meta[b'SLxExperiment'][b'uLoopPars'][b'Points'][b'']

        trans = []
        for raw_trans in raw_translations:
            trans.append([raw_trans[b'dPosX'],
                          raw_trans[b'dPosY'],
                          raw_trans[b'dPosZ']])
        return np.array(trans)

    except KeyError as e:
        logger.error(f"Metadata key not found in ND2 file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while processing ND2 file: {e}")
        raise


def blend(offsets: np.ndarray, pictures: List[np.ndarray], 
          indices: Optional[List[int]] = None, inverts: Optional[List[int]] = None) -> np.ndarray:
    r"""
    Blends a list of volumes tiles together according to specified offsets.

    This function takes a list of volumes (tiles) and their corresponding offsets, 
    then composites them into a single volume. Optionally, some tiles can be inverted 
    along specified axes during the blending process.

    :param offsets: An array of offsets for each image tile.
    :type offsets: np.ndarray
    :param pictures: A list of image tiles to blend.
    :type pictures: List[np.ndarray]
    :param indices: Optional list of indices specifying the order in which to blend the images.
                    If None, the images are blended in their given order.
    :type indices: Optional[List[int]]
    :param inverts: Optional list of axes along which to invert the corresponding image tiles.
                    If None, no inversion is performed.
    :type inverts: Optional[List[int]]
    
    :return: The blended image.
    :rtype: np.ndarray
    """
    if not pictures:
        logger.error("The list of pictures must not be empty.")
    if len(offsets) != len(pictures):
        logger.error("The number of offsets must match the number of pictures.")
    try:
        if indices is None:
            indices = range(0,len(offsets))
        tile_size = np.array(pictures[0].shape)
        origin = np.min(offsets, axis=0)
        newshape = np.ceil(np.abs(np.max(offsets, axis=0)-origin)[[2, 1, 0]] + np.array(tile_size)).astype(np.uint)
        newpic = np.zeros(newshape, dtype=pictures[0].dtype)
        for off, tile, index in zip(offsets, pictures, indices):
            if inverts:
                tile = np.flip(tile, axis=inverts)
            blit = tile
            if pictures[0].dtype == np.dtype('uint8'):
                blitmask = ((blit == 0) * 255).astype(np.uint8)
            elif pictures[0].dtype == np.dtype('uint16'):
                blitmask = ((blit==0)*65535).astype(np.uint16)
            # print(tile.shape, blitmask.shape, transposes, tile_size, int(off[0]-origin[0])+tile_size[2])
            newpic[int(off[2]-origin[2]):int(off[2]-origin[2])+tile_size[0],
                int(off[1]-origin[1]):int(off[1]-origin[1])+tile_size[1],
                int(off[0]-origin[0]):int(off[0]-origin[0])+tile_size[2]] &= blitmask
            newpic[int(off[2]-origin[2]):int(off[2]-origin[2])+tile_size[0],
                int(off[1]-origin[1]):int(off[1]-origin[1])+tile_size[1],
                int(off[0]-origin[0]):int(off[0]-origin[0])+tile_size[2]] |= blit
        return newpic
    except Exception as e:
        logger.error(f"Unexpected error occurred during image blending: {e}")


def blend2(offsets: np.ndarray, pictures: List[np.ndarray]) -> np.ndarray:
    r"""
    Blends a pair of volume tiles together into a 32-bit composite image, according to specified offsets.
    
    Each tile is blit into separate 16 bits of the output image; one tile is placed in the upper 16 bits and 
    the other in the lower 16 bits. This is used during the deduplication process. The blending accounts for 
    the spatial offsets of each tile to correctly align them in the composite output.

    :param offsets: An array of offsets for each image tile, where each row corresponds to a tile's offset.
    :type offsets: np.ndarray
    :param pictures: A list containing exactly two image tiles to blend, each represented as a 3D numpy array.
    :type pictures: List[np.ndarray]

    :return: The blended image as a 32-bit numpy array, where the first and second tiles contribute to separate bit planes.
    :rtype: np.ndarray
    """
    if len(pictures) != 2:
        logger.error("blend2 function requires exactly two image tiles to blend.")
        raise
    if offsets.shape[0] != 2:
        logger.error("Offsets array must have two rows, one for each image tile.")
        raise
    
    try:
        tile_size = np.array(pictures[0].shape)[[2,1,0]]
        origin = np.min(offsets, axis=0)
        newshape = np.ceil(np.abs(np.max(offsets, axis=0)-origin) + np.array(tile_size)).astype(np.uint)
        newpic = np.zeros(newshape, dtype=np.uint32)
        off = offsets[0]
        newpic[int(off[0]-origin[0]):int(off[0]-origin[0])+tile_size[0],
            int(off[1]-origin[1]):int(off[1]-origin[1])+tile_size[1],
            int(off[2]-origin[2]):int(off[2]-origin[2])+tile_size[2]] = np.transpose(pictures[0], axes=[2,1,0])
        off = offsets[1]
        newpic[int(off[0]-origin[0]):int(off[0]-origin[0])+tile_size[0],
            int(off[1]-origin[1]):int(off[1]-origin[1])+tile_size[1],
            int(off[2]-origin[2]):int(off[2]-origin[2])+tile_size[2]] += \
                        np.transpose(np.left_shift(pictures[1].astype(np.uint32),16),axes=[2,1,0])
        return newpic
    
    except Exception as e:
        logger.error(f"An error occurred while blending image tiles: {e}")
        raise

def overlapping(t1: Tuple[int, int, int], 
                t2: Tuple[int, int, int], 
                tile_size: Tuple[int, int, int]) -> bool:
    r"""
    Determines whether the tiles positioned by two transforms will overlap, given the size of the tiles.
    
    This function checks if the spatial positioning of two tiles described by their transform vectors (t1, t2)
    will result in any overlap between them, considering the dimensions of the tiles.

    :param t1: A 3-tuple representing the transform of the first tile (z, x, y offsets).
    :type t1: Tuple[int, int, int]
    :param t2: A 3-tuple representing the transform of the second tile (z, x, y offsets).
    :type t2: Tuple[int, int, int]
    :param tile_size: A 3-tuple representing the size of the tiles (z, x, y dimensions).
    :type tile_size: Tuple[int, int, int]

    :return: True if the tiles overlap, False otherwise.
    :rtype: bool
    """
    if not all(isinstance(n, int) for n in (t1 + t2 + tile_size)):
        raise ValueError("All transform and tile size values must be integers.")
    if not (len(t1) == len(t2) == len(tile_size) == 3):
        raise ValueError("The transform and tile size arguments must be 3-tuples.")
    
    try:
        xs = sorted([t1[0], t2[0]])
        ys = sorted([t1[1], t2[1]])
        zs = sorted([t1[2], t2[2]])
        tz, tx, ty = tile_size
        if xs[0]+tx<xs[1]:
            return False
        if ys[0]+ty<ys[1]:
            return False
        if zs[0]+tz<zs[1]:
            return False
        return True
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while checking tile overlap: {e}")


# Returns a list of overlapping indices in a list of transforms, given a tile_size
def get_tiles_overlaps(transforms, tile_size):
    r"""
    Identifies and returns a set of pairs of indices of transforms that cause their corresponding tiles to overlap.

    Given a list of transforms describing the spatial positioning of tiles and the size of the tiles, 
    this function determines all pairs of tiles that overlap each other.

    :param transforms: A list of 3-tuples where each tuple represents the transform of a tile (z, x, y offsets).
    :type transforms: List[Tuple[int, int, int]]
    :param tile_size: A 3-tuple representing the size of the tiles (z, x, y dimensions).
    :type tile_size: Tuple[int, int, int]

    :return: A set of unique pairs (as tuples) of indices that have overlapping tiles.
    :rtype: Set[Tuple[int, int]]
    """
    if not all(isinstance(t, tuple) and len(t) == 3 for t in transforms):
        raise ValueError("Each transform must be a 3-tuple of integers.")
    if not isinstance(tile_size, tuple) or len(tile_size) != 3:
        raise ValueError("Tile size must be a 3-tuple of integers.")
    
    try:
        overlaps = set()
        for i,oi in enumerate(transforms):
            for j,oj in enumerate(transforms):
                if i==j:
                    continue
                if overlapping(oi,oj, tile_size):
                    overlaps.add((min(i,j), max(i,j)))
        return overlaps

    except Exception as e:
        logger.error(f"An error occurred while identifying tile overlaps: {e}")



# returns a list of overlapping blobs. Each is described as a list of 5 elements:
#               [tile_index1, blob_index1, tile_index2, blob_index2, overlap_area]
def find_overlapping_blobs(tiles: List[np.ndarray], 
                           offsets: List[Tuple[int, int, int]], 
                           progress: bool = False) -> List[List[int]]:
    r"""
    Identifies and returns a list of overlapping blobs between tiles, given their offsets.

    For each pair of overlapping tiles, this function calculates the blend of the two tiles' blob indices and
    determines the overlapping blobs. It returns a list where each element describes an overlap with the
    following format: [tile_index1, blob_index1, tile_index2, blob_index2, overlap_area].

    :param tiles: A list of image volumes (tiles), each represented as a 3D numpy array.
    :type tiles: List[np.ndarray]
    :param offsets: A list of offsets for each tile, specified as (z, x, y) tuples.
    :type offsets: List[Tuple[int, int, int]]
    :param progress: If set to True, prints progress of the operation to stdout. Default is False.
    :type progress: bool

    :return: A list where each element is a list describing an overlapping blob.
    :rtype: List[List[int]]
    """
    if len(tiles) != len(offsets):
        logger.error("The number of tiles must match the number of offsets.")
        raise
    if not all(isinstance(t, np.ndarray) for t in tiles):
        logger.error("All elements in 'tiles' must be numpy arrays.")
        raise
    if not all(isinstance(o, tuple) and len(o) == 3 for o in offsets):
        logger.error("Each offset must be a 3-tuple of integers.")
        raise

    try:
        overlaps = get_tiles_overlaps(offsets, tiles[0].shape)

        overlap_blobs = list()
        for i, (i1, i2) in enumerate(overlaps):
            if progress:
                print(f"Processing overlap {i}/{len(overlaps)}     ", end="\r")
            inter = blend2([offsets[i1], offsets[i2]], [tiles[i1], tiles[i2]])
            values, counts = np.unique(inter, return_counts=True)
            for i, count in zip(values.tolist(), counts.tolist()):
                ind1 = i % 65536
                ind2 = i//65536
                if i < 65536:
                    continue
                if i % 65536 == 0:
                    continue
                if count > 0:
                    overlap_blobs.append([i1, ind1, i2, ind2, count])
        return overlap_blobs
    except Exception as e:
        logger.error(f"An error occurred while finding overlapping blobs: {e}")
        raise


def apply_new_ids(img: np.ndarray, new_ids: Dict[Tuple[int, int], int], tile_id: int) -> np.ndarray:
    r"""
    Applies new IDs to blobs in an image based on a mapping from old IDs to new IDs.

    This function creates a new image array where each blob's ID is replaced with a new ID according to a
    provided mapping. The mapping is a dictionary where keys are tuples of the form (tile_id, old_blob_id),
    and values are the new blob IDs.

    :param img: The input image with blob IDs to be replaced.
    :type img: np.ndarray
    :param new_ids: A dictionary mapping (tile_id, old_blob_id) to new_blob_id.
    :type new_ids: Dict[Tuple[int, int], int]
    :param tile_id: The identifier of the current tile being processed.
    :type tile_id: int

    :return: A new image with updated blob IDs.
    :rtype: np.ndarray
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2:
        logger.error("The input image must be a 2D numpy array.")
        raise
    if not new_ids:
        logger.error("The new_ids mapping cannot be empty.")
        raise
    
    try:
        new_img = np.zeros(img.shape, dtype=np.uint16)
        for blob_i, blob_size in enumerate(np.bincount((img.flatten()))):
            if blob_size > 0 and blob_i != 0:
                nid = new_ids[(tile_id, blob_i)]
                new_img[img == blob_i] = nid
        return new_img
    
    except Exception as e:
        logger.error(f"An error occurred while applying new IDs to blobs: {e}")
        raise


def deduplicate_blob_ids(tiles: List[np.ndarray], offsets: np.ndarray, progress: bool = False) -> Dict[Tuple[int, int], int]:
    r"""
    Generates a mapping from (tile_id, blob_id) pairs to a new unique ID for each blob, merging overlapping blobs.

    This function identifies blobs across tiles that should be merged based on overlaps and assigns a new unique ID to
    the merged set of blobs. Blobs that do not overlap with others will also receive a unique new ID.

    :param tiles: A list of image tiles with blob IDs to be deduplicated.
    :type tiles: List[np.ndarray]
    :param offsets: An array of offsets indicating the position of each tile.
    :type offsets: np.ndarray
    :param progress: Whether to display progress information.
    :type progress: bool

    :return: A dictionary mapping each (tile_id, blob_id) to a new unique ID.
    :rtype: Dict[Tuple[int, int], int]
    """
    if not isinstance(tiles, list) or not all(isinstance(tile, np.ndarray) for tile in tiles):
        logger.error("Tiles must be a list of numpy arrays.")
        raise
    if not isinstance(offsets, np.ndarray):
        logger.error("Offsets must be a numpy array.")
        raise

    try:
        new_id = dict()  # (tile_id, blob_id) -> 16 bits UID
        # For each blob, in each other tile, which blob is the most likely candidate for a merge?
        merge_candidate_per_tile = defaultdict(lambda: defaultdict(lambda: (-1, 0)))

        all_blobs = get_all_blobs(tiles)
        overlap_blobs = find_overlapping_blobs(tiles, offsets, progress)

        for k in range(8):
            for tile1, ind1, tile2, ind2, v in overlap_blobs:
                cind, coverlap = merge_candidate_per_tile[(tile1, ind1)][tile2]
                cind2, coverlap2 = merge_candidate_per_tile[(tile2, ind2)][tile1]

                if v > coverlap and v > coverlap2:
                    merge_candidate_per_tile[(tile1, ind1)][tile2] = (ind2, v)
                    merge_candidate_per_tile[(tile2, ind2)][tile1] = (ind1, v)
                    try:
                        del merge_candidate_per_tile[(tile2, cind)][tile1]
                    except KeyError:
                        pass
                    try:
                        del merge_candidate_per_tile[(tile1, cind2)][tile2]
                    except KeyError:
                        pass

        mergesets = list()
        for k, v in merge_candidate_per_tile.items():
            blobs = set()
            blobs.add(k)
            for k2, v2 in v.items():
                blobs.add((k2, v2[0]))
            added = False
            for s in mergesets:
                for b in blobs:
                    if b in s:
                        s.union(blobs)
                        added = True
                        break
            if not added:
                mergesets.append(blobs)

        newi = 1
        for s in mergesets:
            for blob in s:
                new_id[blob] = newi
            newi += 1

        for (tileind, blobind) in all_blobs:
            if (tileind, blobind) not in new_id.keys():
                new_id[(tileind, blobind)] = newi
                #         new_id3[(tileind, blobind)]=0
                newi += 1
        return new_id
    
    except Exception as e:
        logger.error(f"An error occurred during blob deduplication: {e}")
        raise


def get_all_blobs(tiles: List[np.ndarray]) -> Set[Tuple[int, int]]:
    r"""
    Extracts all unique blob identifiers from a list of tiles.

    This function iterates through a list of images (tiles) and extracts all unique
    (tile index, blob identifier) pairs present across the tiles.

    :param tiles: A list of image tiles containing blob identifiers.
    :type tiles: List[np.ndarray]

    :return: A set containing all unique (tile index, blob identifier) pairs.
    :rtype: Set[Tuple[int, int]]
    """
    if not isinstance(tiles, list) or not all(isinstance(tile, np.ndarray) for tile in tiles):
        logger.error("Tiles must be a list of numpy arrays.")
        raise
    
    try:
        all_blobs = set()
        for i, img in enumerate(tiles):
            for ind, v in enumerate(np.bincount((img.flatten()))):
                if v > 0:
                    all_blobs.add((i, ind))
        return all_blobs
    
    except Exception as e:
        logger.error(f"An error occurred while extracting blob identifiers: {e}")
        raise

def make_stitched_dedup_h5(target_file: str, tiles: List[np.ndarray], trans: List[Tuple[float, float, float]], new_ids: dict) -> List[np.ndarray]:
    r"""
    Creates a BigDataViewer (BDV) H5 file with deduplicated and stitched tiles.

    This function takes a list of image tiles, their transformations (translation offsets), and a dictionary of new IDs
    for deduplication. It stitches these tiles into a single volume, remaps their blob IDs according to the new_ids,
    and saves the volume as an H5 file readable by the BigDataViewer.

    :param target_file: The path to the output H5 file.
    :type target_file: str
    :param tiles: A list of 3D image tiles (numpy arrays).
    :type tiles: List[np.ndarray]
    :param trans: A list of translation offsets for each tile.
    :type trans: List[Tuple[float, float, float]]
    :param new_ids: A dictionary mapping old blob IDs to new unique blob IDs.
    :type new_ids: dict

    :return: A list of new image tiles with deduplicated IDs.
    :rtype: List[np.ndarray]
    """
    cur_dir = os.getcwd()

    if not os.path.isdir(cur_dir):
        logger.error(f"The directory {cur_dir} does not exist.")
        raise

    if not isinstance(new_ids, dict):
        logger.error("new_ids must be a dictionary.")
        raise

    if not all(isinstance(tile, np.ndarray) and tile.ndim == 3 for tile in tiles):
        logger.error("Each tile must be a 3D numpy array.")
        raise

    if not all(isinstance(t, tuple) and len(t) == 3 for t in trans):
        logger.error("Each translation offset must be a 3-tuple.")
        raise
    
    try:
        new_tiles=list()
        unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                            (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                            (0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)
        if os.path.exists(f"{target_file}"):
            os.remove(f"{target_file}")
        bdv_writer = npy2bdv.BdvWriter(f"{target_file}", ntiles=len(trans))
        dbg=list()
        for i in range(len(trans)):
            fn = f"seg_fov{i}.tif"
            print(f"Adding {fn}")
            orig_img = tifffile.imread(f"{cur_dir}/{fn}")
            new_img = np.zeros(orig_img.shape, dtype=np.uint16)
            for blob_i, blob_size in enumerate(np.bincount((orig_img.flatten()))):
                if blob_size>0 and blob_i!=0:
                    nid = new_ids[(i, blob_i)]
                    new_img[orig_img==blob_i] = nid
            new_tiles.append(new_img)
            new_img = np.transpose(new_img, axes=[0,2,1])
            affine_matrix = unit_matrix
            affine_matrix[0,3] = trans[i][0]
            affine_matrix[1,3] = trans[i][1]
            affine_matrix[2,3] = trans[i][2]
            bdv_writer.append_view(new_img, time=0,
                                tile=i,
                                m_affine=affine_matrix,
                                name_affine=f"tile {i} translation")
        bdv_writer.write_xml()
        bdv_writer.close()
        return new_tiles
    except Exception as e:
        logger.error(f"An error occurred while make stitched dedup h5 : {e}")
        raise


def make_stitched_h5(target_file: str, tiles: List[np.ndarray], transforms: List[Tuple[float, float, float]]) -> None:
    r"""
    Creates a BigDataViewer (BDV) H5 file with stitched tiles.

    This function stitches a list of image tiles together using the provided transformations and 
    saves the volume as an H5 file readable by BigDataViewer.

    :param target_file: The path to the output H5 file.
    :type target_file: str
    :param tiles: A list of 3D image tiles (numpy arrays).
    :type tiles: List[np.ndarray]
    :param transforms: A list of translation offsets for each tile.
    :type transforms: List[Tuple[float, float, float]]
    """
    # Check the target directory
    target_dir = os.path.dirname(target_file)
    if not os.path.exists(target_dir):
        logger.error(f"The directory {target_dir} does not exist for the target file.")
        raise

    if not isinstance(tiles, list) or not all(isinstance(tile, np.ndarray) for tile in tiles):
        logger.error("tiles must be a list of numpy ndarrays.")
        raise

    if not isinstance(transforms, list) or not all(isinstance(trans, tuple) and len(trans) == 3 for trans in transforms):
        logger.error("transforms must be a list of 3-tuples.")
        raise

    if os.path.exists(target_file):
        os.remove(target_file)

    try:
        unit_matrix = np.array(((1.0, 0.0, 0.0, 0.0), # change the 4. value for x_translation (px)
                            (0.0, 1.0, 0.0, 0.0), # change the 4. value for y_translation (px)
                            (0.0, 0.0, 1.0, 0.0)))# change the 4. value for z_translation (px)
        if os.path.exists(f"{target_file}"):
            os.remove(f"{target_file}")
        bdv_writer = npy2bdv.BdvWriter(f"{target_file}", ntiles=len(transforms))

        i=0
        for orig_img, trans in zip(tiles, transforms):
            fn = f"seg_fov{i}.tif"
            print(f"Adding {fn}")
            affine_matrix = unit_matrix
            affine_matrix[0,3] = trans[0]
            affine_matrix[1,3] = trans[1]
            affine_matrix[2,3] = trans[2]
            bdv_writer.append_view(orig_img, time=0,
                                tile=i,
                                m_affine=affine_matrix,
                                name_affine=f"tile {i} translation")
            i+=1
        bdv_writer.write_xml()
        bdv_writer.close()
    
    except Exception as e:
        logger.error(f"An error occurred while make stitched h5 : {e}")
        raise


# Calculates the surface where cells start being visibles, used in physical stitching experiments
def get_contact_surface(img: np.ndarray, direction: int = 1, thresh: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    r"""
    Calculates the surface where cells start being visible in an image stack.

    The function moves through the stack in the specified direction, 
    identifying the first z-slice where the intensity is above a calculated threshold, 
    which can also be provided.

    :param img: 3D image stack.
    :type img: np.ndarray
    :param direction: Direction to process the stack; 1 from bottom, -1 from top.
    :type direction: int
    :param thresh: Optional threshold value(s); if not provided, it's calculated from the image.
    :type thresh: Optional[np.ndarray]

    :return: Tuple containing the image surface, the indices of the surface, and the threshold used.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    if img.ndim != 3:
        logger.error("The input image must be a 3D numpy array.")
        raise

    if direction not in (-1, 1):
        logger.error("The direction must be either 1 (from bottom) or -1 (from top).")
        raise
    try:
        if thresh is None:
            per = np.percentile(img, 99, axis=2)
            thresh = per[:, :] / 3
        minsurf = np.zeros((img.shape[:2]))
        argsmin = np.zeros((img.shape[:2]))
        if direction == 1:
            d1, d2 = 0, img.shape[2]
        elif direction == -1:
            d1, d2 = img.shape[2] - 1, -1
        for z in range(d1, d2, direction):
            print(f"{z}/{img.shape[2]}     ", end="\r")
            cond1 = img[:, :, z] > thresh
            cond1 = dilate(cond1, np.ones((3, 3)), iterations=5)
            cond2 = argsmin == 0
            cond = cond1 & cond2

            minsurf[cond] = img[:, :, z][cond]
            argsmin[cond] = (np.ones(argsmin.shape) * z)[cond]
        return minsurf, argsmin, thresh
    
    except Exception as e:
        logger.error(f"An error occurred while getting contact surface : {e}")
        raise


def get_top_surface(img: np.ndarray, thresh: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Identifies the top surface of the visible cells in an image stack.

    This function is a wrapper for `get_contact_surface` that calculates the surface 
    starting from the top of the stack.

    :param img: 3D image stack.
    :type img: np.ndarray
    :param thresh: Optional threshold value(s); if not provided, it's calculated from the image.
    :type thresh: Optional[np.ndarray]

    :return: Tuple containing the image surface, the indices of the surface, and the threshold used.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    return get_contact_surface(img, -1, thresh)


def get_bottom_surface(img: np.ndarray, thresh: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Identifies the bottom surface of the visible cells in an image stack.

    This function is a wrapper for `get_contact_surface` that calculates the surface 
    starting from the bottom of the stack.

    :param img: 3D image stack.
    :type img: np.ndarray
    :param thresh: Optional threshold value(s); if not provided, it's calculated from the image.
    :type thresh: Optional[np.ndarray]

    :return: Tuple containing the image surface, the indices of the surface, and the threshold used.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    return get_contact_surface(img, 1, thresh)