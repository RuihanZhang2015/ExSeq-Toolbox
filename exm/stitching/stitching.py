import os
import xml.etree.ElementTree
from collections import defaultdict
from scipy.ndimage import binary_dilation as dilate

import npy2bdv
import numpy as np
from nd2reader import ND2Reader
import tifffile


def get_offsets(filename):
    # Extracts the offset from a BDV/H5 XML file and returns them as a (N,3) array in the ((X,Y,Z),...) order
    tree = xml.etree.ElementTree.parse(filename)
    root = tree.getroot()
    vtrans = list()
    for registration_tag in root.findall('./ViewRegistrations/ViewRegistration'):
        tot_mat = np.eye(4, 4)
        for view_transform in registration_tag.findall('ViewTransform'):
            affine_transform = view_transform.find('affine')
            mat = np.array([float(a) for a in affine_transform.text.split(" ") if a != ""] + [0, 0, 0, 1]).reshape((4, 4))
            tot_mat = np.matmul(tot_mat, mat)
        vtrans.append(tot_mat)

    def transform_to_translate(m):
        m[0, :] = m[0, :] / m[0][0]
        m[1, :] = m[1, :] / m[1][1]
        m[2, :] = m[2, :] / m[2][2]
        return m[:-1, -1]

    trans = [transform_to_translate(vt).astype(np.int64) for vt in vtrans]
    return np.stack(trans)


def get_offsets_nd2(filename):
    # Extracts the offset from a ND2 file and returns them as a (N,3) array in the ((X,Y,Z),...) order
    images = ND2Reader(filename)
    meta = images._parser._raw_metadata.image_metadata
    raw_translations = meta[b'SLxExperiment'][b'uLoopPars'][b'Points'][b'']

    trans = list()
    for raw_trans in raw_translations:
        trans.append([raw_trans[b'dPosX'],
                      raw_trans[b'dPosY'],
                      raw_trans[b'dPosZ']])
    return trans


def blend(offsets, pictures, indices=None, inverts=None):
    # Blends a list of tiles together according to offsets
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


def blend2(offsets, pictures):
    # TODO: fix (remove) the transposes here
    # Blends 2 tiles of 16 bpp together into a 32 bpp. One tile is blit on the upper byte and one on the bottom byte
    # This is used during the deduplication process
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


def overlapping(t1, t2, tile_size):
    # Checks if two transforms, given a tile size, will make tiles overlap
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


# Returns a list of overlapping indices in a list of transforms, given a tile_size
def get_tiles_overlaps(transforms, tile_size):
    overlaps = set()
    for i,oi in enumerate(transforms):
        for j,oj in enumerate(transforms):
            if i==j:
                continue
            if overlapping(oi,oj, tile_size):
                overlaps.add((min(i,j), max(i,j)))
    return overlaps


# returns a list of overlapping blobs. Each is described as a list of 5 elements:
#               [tile_index1, blob_index1, tile_index2, blob_index2, overlap_area]
def find_overlapping_blobs(tiles, offsets, progress=False):
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


def apply_new_ids(img, new_ids, tile_id):
    new_img = np.zeros(img.shape, dtype=np.uint16)
    for blob_i, blob_size in enumerate(np.bincount((img.flatten()))):
        if blob_size > 0 and blob_i != 0:
            nid = new_ids[(tile_id, blob_i)]
            new_img[img == blob_i] = nid
    return new_img


def deduplicate_blob_ids(tiles, offsets, progress=False):
    # Produces a map of a tile_id, blob_id pair into a new ID, unique over the dataset,
    # merging overlapping blobs from different tiles

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


def get_all_blobs(tiles):
    # Makes a list of all the existing tile(id, blob-id) pairs
    all_blobs = set()
    for i, img in enumerate(tiles):
        for ind, v in enumerate(np.bincount((img.flatten()))):
            if v > 0:
                all_blobs.add((i, ind))
    return all_blobs


def make_stitched_dedup_h5(target_file, tiles, trans, new_ids):
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


def make_stitched_h5(target_file, tiles, transforms):
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


# Calculates the surface where cells start being visibles, used in physical stitching experiments
def get_contact_surface(img, direction=1, thresh=None):
    # direction=1  => from bottom
    # direction=-1  => from top
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


def get_top_surface(img, thresh=None):
    return get_contact_surface(img, -1, thresh)


def get_bottom_surface(img, thresh=None):
    return get_contact_surface(img, 1, thresh)