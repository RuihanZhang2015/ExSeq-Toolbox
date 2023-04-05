import random
import tifffile
import numpy as np
from nd2reader import ND2Reader
import h5py
import npy2bdv
import struct
import glymur
import pathlib
from natsort import natsorted # pip install natsort
from scipy import ndimage

import exm.stitching.stitching as stitching

"""# TODO: Write examples:
    - Load from ND2
    - Pre-stitch into H5BDV
    - Intensity scaling
    - Build volume from intensities
    - Build volume from segmented files
    - Centroid detection 
    
    Optional:
    - Segment with CysGAN
    - Load from JP2
    - Save to JP2
"""
# TODO: Write tests.
# WIP: Find a way to preserve pixel scale info into H5/BDV
#   - show_slice has the canonical way of doing it.
#   - TODO: Scaling hacks have to be removed from other functions
# TODO: Batch save to JP2/TIFF from Tileset
# TODO: Demonstrate reconstructon from JP2 + XML
#        cf: Notebook 2023-01-25 Reconstruct slices from JP2 files
# TODO: Ability to output XML alone? Maybe custom pickle format?
# TODO: Init a tileset without a ND2 file:
#    - with a XML/H5 pair
#    - with a bunch of JP2 + .npy offsets files
# TODO: Document all XYZ orders (scale, voxel_size, original_size, etc.)


"""
    Some tips to avoid inversions/transpositions/axis inversions:
    
      - Translations and offsets, valumes in µm are typically given in the X,Y,X order
      - Images are stored in C-order which means the X axis is the most contiguous, followed by Y and then Z. This 
        result in pixels accessed in the [z,y,x] order.
      - Image shapes are therefore specified in [z,y,x] order
      - For some reason, the encoder X translation is inverted w.r.t the +x axis of individual images in the ND2 files
      - For now, H5BDV files do not contain µm information. All transforms are expressed in pixels
"""


class Tileset:
    def __init__(self, voxel_size, orig_size=None):
        self.tiles = list()
        self.voxel_size = voxel_size
        self.original_xyz_size = orig_size
        self.nd2=None

    def init_from_jp2(self, files_list, offsets_file, downscale=[1, 1, 1]):
        # initializes the tileset from a list of JP2 files, each representing a single
        # tile, and an offset files, which is .npy file storing a 2D array. The first row
        # of this array gives the original pixel size of the tile and the other rows the
        # offset of the tiles.

        a = np.load(offsets_file)
        self.original_xyz_size = a[0].astype(int)
        offsets = a[1:]
        offsets[:, 0] = -offsets[:,0]
        offsets = offsets - np.min(offsets, axis=0)

        for i,off in enumerate(offsets):
            print(f"{i + 1}/{len(offsets)} {' ' * 50}", end="\r")
            t = Tile(i)
            t.offset = off
            t.from_jp2(files_list[i], downscale=downscale)
            self.tiles.append(t)

    def init_from_nd2(self, nd2):
        # initializes the tileset from a ND2 file. As these files can be slow to load
        # it does not actually read the content of the tiles, but only reads metadata
        # and will lazily load the necessary information on calls to `load_fov` or
        # `preview_nd2`

        if type(nd2) is str:
            nd2 = ND2Reader(nd2)
        self.nd2=nd2
        # Extract offsets from ND2 file
        meta = self.nd2._parser._raw_metadata.image_metadata
        raw_translations = meta[b'SLxExperiment'][b'uLoopPars'][b'Points'][b'']
        offsets = list()
        for i,raw_trans in enumerate(raw_translations):
            offsets.append([raw_trans[b'dPosX'],
                                 raw_trans[b'dPosY'],
                                 raw_trans[b'dPosZ']])
        offsets = np.array(offsets)
        offsets[:,0] = -offsets[:,0]
        offsets = offsets - np.min(offsets, axis=0)

        for i,off in enumerate(offsets):
            t = Tile(i)
            t.offset = off
            self.tiles.append(t)

        self.original_xyz_size = [self.nd2.sizes["x"],self.nd2.sizes["y"],self.nd2.sizes["z"]]

    def init_from_bdv(self, filename):
        # Loads a tileset from a BDV H5 (BigDataViewer format). The argument should be either the .h5 or the .xml file
        h5bdv = pathlib.Path(filename).with_suffix(".h5")
        self.xml_file = pathlib.Path(filename).with_suffix(".xml")
        offsets = stitching.get_offsets(self.xml_file)
        offsets = offsets - np.min(offsets, axis=0)
        file = h5py.File(h5bdv, 'r')

        tilesnum = len(file['t00000'].keys())
        self.tiles.clear()
        for i in range(tilesnum):
            t = Tile(i)
            t.offset = offsets[i] * self.voxel_size
            t.img = file['t00000'][f's{i:02d}']['0']['cells'].astype(np.uint16)[:]
            self.tiles.append(t)

        # TODO: check if we don't need to change the order of axes here
        self.original_xyz_size = [self.tiles[0].img.shape[2],
                                  self.tiles[0].img.shape[1],
                                  self.tiles[0].img.shape[0]]

    def init_from_h5(self, filename, downscale=[1,1,1], progress=False):
        # Loads a tileset from a non-BDV H5, will expect transforms from another source
        file = h5py.File(filename, 'r')
        keys = []
        file.visit(lambda key: keys.append(key) if isinstance(file[key], h5py.Dataset) else None)

        self.tiles = list()
        for i,k in enumerate(natsorted(keys)):
            if progress:
                print(f"Tile #{i}")
            t = Tile(i)
            t.img = file[k].astype(np.uint16)[::downscale[2], ::downscale[1], ::downscale[0]]
            self.tiles.append(t)

        self.original_xyz_size = [file[keys[0]].shape[2],
                                  file[keys[0]].shape[1],
                                  file[keys[0]].shape[0]]

    def init_from_tiff_files(self, filelist, downscale=[1,1,1], progress=False):
        # Init tiles from a list of TIFF files but will not initiialize offsets
        self.tiles = list()
        for i,fn in enumerate(filelist):
            if progress:
                print(f"Tile #{i}")
            t = Tile(i)
            orig = tifffile.imread(fn)
            t.img = orig.astype(np.uint16)[::downscale[2], ::downscale[1], ::downscale[0]]
            self.tiles.append(t)
        self.original_xyz_size = [orig.shape[2],
                                  orig.shape[1],
                                  orig.shape[0]]

    def scale_offset(self, scale=(1,1,1)):
        # Scales all offsets by a factor. Expects a X, Y, Z order in the argument
        for i in range(len(self.tiles)):
            self.tiles[i].offset = self.tiles[i].offset * scale

    def __getitem__(self, i):
        # One can access to tiles using the [] operator
        return self.tiles[i]

    def get_slice(self, fovnum, z):
        # Returns a specific Z-slice from a specific FOV. That's a fast function that does not load the whole tile
        self.nd2.bundle_axes = "yx"
        self.nd2._default_coords["v"] = fovnum
        self.nd2.iter_axes = "z"
        return np.array(self.nd2[z])

    def preview_nd2(self, z_layer=0, down_sample='auto'):
        """
            Previews a slice of a stitched version of the ND2 without loading all the tiles
        """
        pictures = [self.get_slice(fovnum, z_layer + self.tiles[fovnum].offset[2]/self.voxel_size[2]) for fovnum in range(len(self.tiles))]
        offsets = [t.offset[:2] / self.voxel_size[:2] for t in self.tiles]
        tile_size = np.array(pictures[0].shape)
        origin = np.min(offsets, axis=0)
        newshape = np.ceil(np.abs(np.max(offsets, axis=0) - origin) + np.array(tile_size)).astype(np.uint)
        newpic = np.zeros(newshape[::-1], dtype=np.uint16)
        div = np.zeros(newshape[::-1], dtype=np.uint8)
        for off, tile in zip(offsets, pictures):
            newpic[int(off[1] - origin[1]):int(off[1] - origin[1]) + tile_size[0],
                   int(off[0] - origin[0]):int(off[0] - origin[0]) + tile_size[1]] += tile
            div[int(off[1] - origin[1]):int(off[1] - origin[1]) + tile_size[0],
                   int(off[0] - origin[0]):int(off[0] - origin[0]) + tile_size[1]] += np.ones(tile.shape, dtype=np.uint8)
        newpic = newpic/div
        if down_sample == "auto":
            # Finding downsampling factors to have a max dim of ~1500
            down_sample = int(np.max(newpic.shape)/1500)
        return np.clip(newpic[::down_sample, ::down_sample], 0, 255).astype(np.uint8)

    # def load_tileset_h5bdv(self, filename):
    #     h5bdv = filename
    #     xml_file = pathlib.Path(h5bdv).with_suffix(".xml")
    #     offsets = exmfs.stitching.get_offsets(xml_file)
    #     file = h5py.File(h5bdv, 'r')
    #
    #     orig_size = np.array([self.nd2.sizes["z"], self.nd2.sizes["y"], self.nd2.sizes["x"]])
    #     tile_size = file['t00000'][f's00']['0']['cells'].shape
    #     scale_factor = tile_size / orig_size
    #
    #     tilesnum = len(file['t00000'].keys())
    #     for i in range(tilesnum):
    #         self.tiles[i].img = file['t00000'][f's{i:02d}']['0']['cells'].astype(np.uint16)
    #         self.tiles[i].offset = offsets[i] * self.voxel_size / scale_factor[::-1]

    def get_centroids(self):
        # Returns a list of centroids as an array of XYZ coordinates. It can take a while. As this function internally
        # calls `produce_output_volume()` it is recommended to scale down the tiles first

        print("Produce output volume")
        vol = self.produce_output_volume()
        # Find centers, removing ID #0, which would be the centroind for the black pixels
        print("produce bincount")
        a = np.bincount(vol.flatten())

        print("calculate centroids")
        centers = ndimage.measurements.center_of_mass(vol, vol, np.nonzero(a))
        centers = np.array(centers)[:, 0, :]
        return centers[1:,[2,1,0]]

    def create_blank_h5bdv(self, target_file, tile_size=None, dtype=np.int16, noise_scale=200.0):
        """
        Creates a H5/BDV file pair. It is hard to add new tiles to an existing file so instead the H5BDV files are
        created with all their tiles filled with random noise. Why noise instead of zeros? Well Fiji's BigStitcher does
        not like it when some tiles are totally empty and saturates the other tiles. That's why we fill it with noise.
        To deactivate that put noise_scale at 0.0

        :param target_file str: Must be the path to the desired H5 file. An XML file will be created at the same place
                                with the same name and the extension changed.

        :param tile_size (int, int, int): The size we want the tiles to be

        :param dtype (np.dtype): The type to initialize the arrays with

        :param noise_cale (float): The intensity of the white noise the tiles are initialized with

        """
        # TODO: investigate how to put pixel size info in BDV files
        # TODO: check if the method to append views within npy2bdv would not work better:
        #       https: // github.com / nvladimus / npy2bdv / blob / master / docs / examples / examples_h5writing.ipynb

        bdv_writer = npy2bdv.BdvWriter(f"{target_file}", ntiles=len(self.tiles), overwrite=True)

        orig_size = np.array(self.original_xyz_size).astype(int)
        if tile_size is None:
            tile_size = orig_size[::-1]
        scale_factor = tile_size[::-1]/orig_size

        for tile in self.tiles:
            unit_matrix = np.eye(4)[:3, :]
            offset = tile.offset
            offset[0] = -offset[0]
            unit_matrix[:,3] = offset*scale_factor/self.voxel_size
            # unit_matrix[0, 0] = scale_factor[2]
            # unit_matrix[1, 1] = scale_factor[1]
            # unit_matrix[2, 2] = scale_factor[0]
            bdv_writer.append_view(#np.zeros(tile_size, dtype=dtype),
                                   (np.random.random(tile_size)*noise_scale).astype(dtype),
                                   time=0,
                                   tile=tile.fovnum,
                                   m_affine=unit_matrix,
                                   name_affine=f"tile {tile.fovnum} translation")
        bdv_writer.write_xml()
        bdv_writer.close()

    def load_tile(self, fovnums, downscale=[1,1,1], method="fast"):
        # Loads a tile from the ND2 file. Before that call, tiles exist but are placeholders without image data
        # Method can be "fast" or "safe". The fast method makes the loading faster but makes some assumptions about the
        # data format
        #
        # TODO: implement the "safe" method
        if type(fovnums) is int:
            fovnums = [fovnums]
        for fovnum in fovnums:
            self.tiles[fovnum].fast_nd2_read_downscale(self.nd2, downscale)

    def load_all(self, downscale=[1,1,1], method="fast"):
        # Loads all tiles from the ND2 file. Before that call, tiles exist but are placeholders without image data
        # Method can be "fast" or "safe". The fast method makes the loading faster but makes some assumptions about the
        # data format
        # Note that the downscale parameter is in the axis order of the tiles, typically ZYX
        #
        # TODO: implement the "safe" method
        for i,t in enumerate(self.tiles):
            print(f"                   Loading tile {i+1}/{len(self.tiles)} {' '*50}", end="\r")
            t.fast_nd2_read_downscale(self.nd2, downscale)

    def find_intensity_scale(self, progress=False):
        """
        Finds the 1st and 99th percentile of intensity values in the dataset. Can be long.

        """
        ts = [t for t in self.tiles if t.img is not None]
        count = 0
        percentiles = list()
        for i,t in enumerate(ts):
            if progress:
                print(f"{i + 1}/{len(ts)} {' ' * 50}", end="\r")
            percentiles.append(t.find_intensity_scale())
        return np.mean(percentiles, axis=0)

    def scale_intensity(self, p1p99=None, progress=False):
        """
        Scales the intensities over the whole dataset and convert tiles to 8bpp.

        :param p1p99 (int, int): lowest and highest value between which to scale. If None, the function will
                                 calculate the 1st and 99th percentiles of intensities and use these.

        :param progress: If True displays the progress of the percentile calculation (which is long)
        """
        if p1p99 is None:
            p1p99 = self.find_intensity_scale(progress=progress)
        for t in self.tiles:
            t.scale_intensity(p1p99[0], p1p99[1])

    def write_into_h5bdv(self, filename, dtype=np.int16):
        """ Exports the tileset as a H5BDV file format. It will create a .h5 file containing the image information
            and a XML file containing the offsets."""
        # TODO: investigate how to put pixel size info in BDV files
        # TODO: check if the method to append views within npy2bdv would not work better:
        #       https: // github.com / nvladimus / npy2bdv / blob / master / docs / examples / examples_h5writing.ipynb

        bdv_writer = npy2bdv.BdvWriter(f"{filename}", ntiles=len(self.tiles), overwrite=True)

        orig_size = np.array(self.original_xyz_size).astype(int)
        tile_size = self.tiles[0].img.shape
        scale_factor = tile_size[::-1]/orig_size

        for tile in self.tiles:
            unit_matrix = np.eye(4)[:3, :]
            offset = tile.offset
            unit_matrix[:,3] = offset*scale_factor/self.voxel_size
            bdv_writer.append_view(tile.img.astype(dtype),
                                   time=0,
                                   tile=tile.fovnum,
                                   m_affine=unit_matrix,
                                   name_affine=f"tile {tile.fovnum} translation")
        bdv_writer.write_xml()
        bdv_writer.close()
        #
        # for t in self.tiles:
        #     t.write_into_h5bdv(filename)

    def show_slice(self, zslice, down_sample="auto"):
        """
        Shows a single Z-slice of the stitched tileset.

        :param zslice int: z-index to display
        :param down_sample [int, int] or "auto": integer factor by which to downsample each dimension of the result. If
                                                 set to "auto", will make the biggest dimension as close as possible to
                                                 1500

        """
        pictures = list()
        scale = np.array(self.original_xyz_size) / np.array(self.tiles[0].img.shape)[[2,1,0]]
        offsets = np.array([t.offset / (np.array(self.voxel_size) * scale) for t in self.tiles])
        origin = np.min(offsets, axis=0)

        for off, t in zip(offsets, self.tiles):
            try:
                pictures.append(t.img[int(zslice + off[2] - origin[2])])
            except IndexError:
                pictures.append(np.zeros(t.img[0].shape, dtype = t.img[0].dtype))

        tile_size = np.array(pictures[0].shape)
        newshape = np.ceil(np.abs(np.max(offsets[:,:2], axis=0) - origin[:2]) + np.array(tile_size)).astype(np.uint)
        newpic = np.zeros(newshape[::-1], dtype=np.uint16)
        div = np.zeros(newshape[::-1], dtype=np.uint8)
        for off, tile in zip(offsets, pictures):
            newpic[int(off[1] - origin[1]):int(off[1] - origin[1]) + tile_size[0],
                   int(off[0] - origin[0]):int(off[0] - origin[0]) + tile_size[1]] += tile
            div[int(off[1] - origin[1]):int(off[1] - origin[1]) + tile_size[0],
                   int(off[0] - origin[0]):int(off[0] - origin[0]) + tile_size[1]] += np.ones(tile.shape, dtype=np.uint8)
        div[div==0]=1 # Remove a warning about 0./0. divisions occuring
        newpic = newpic/div
        if down_sample == "auto":
            # Finding downsampling factors to have a max dim of ~1500
            down_sample = max(int(np.max(newpic.shape)/1500), 1)
        # return np.clip(newpic[::down_sample, ::down_sample], 0, 255).astype(np.uint8)
        return newpic[::down_sample, ::down_sample].astype(int)

    def update_offsets(self, xml_file, scale_factor=None):
        """
        Replaces the current offsets by those in a specific XML file. Does not touch the pixels information.

        :param xml_file str: the file to read new offsets from
        :param scale_factor: the vector by which to multiply offsets if necessary. Defaults to None. Usually unnecessary
        """
        if scale_factor is None:
            scale_factor = np.array(self.original_xyz_size) / np.array(self.tiles[0].img.shape)[[2,1,0]]
        offsets = np.array(stitching.get_offsets(xml_file))*(self.voxel_size*np.array(scale_factor))
        for t,o in zip(self, offsets):
            t.offset=o

    def produce_output_volume(self):
        # Creates a stitched volume. Careful, if you never downsampled the tiles it easily fills up all memory.
        scale = np.array(self.original_xyz_size) / np.array(self.tiles[0].img.shape)[[2, 1, 0]]
        return stitching.blend(
            [t.offset/(np.array(self.voxel_size)*scale) for t in self.tiles],
            [t.img for t in self.tiles]
        )

    def local_to_global(self, coords):
        offsets = np.array([t.offset for t in self.tiles])
        origin = np.min(offsets, axis=0)
        results = coords[:, 1:] + offsets[coords[:, 0].astype(int)] - origin
        return results

    def dedup_segmentation_ids(self, progress=False):
        # If the tileset is a set of segmented FOVs, this function replaces the tiles IDs by identifiers that are
        # unique accross the dataset
        #
        # TODO: check if the number of IDs fits in uint16 and if not either make a warning or switch to uint32
        #  automatically

        scale = np.array(self.voxel_size) * np.array(self.original_xyz_size) / np.array(self.tiles[0].img.shape)[[2, 1, 0]]
        new_ids = stitching.deduplicate_blob_ids([t.img for t in self.tiles],
                                                 [t.offset/scale for t in self.tiles],
                                                 progress)
        luts = list()
        lut_max = np.max(np.array(list([k[1] for k in new_ids.keys()])))
        for i in range(len(self.tiles)):
            luts.append(np.zeros(lut_max+1, np.uint16))
        for k,v in new_ids.items():
            luts[k[0]][k[1]]=v
        for i in range(len(self.tiles)):
            luts[i][0] = 0
        for i, tile in enumerate(self.tiles):
            print(i)
            self.tiles[i].img = luts[i][tile.img]
        return luts, new_ids


class Tile:
    """
        A Tile represents a FOV in the dataset.
        It contains the FOV number in the dataset, methods to retrieve the original image and to process it, with the
        ability to cache various intermediate results.

    """

    def __init__(self, fovnum):
        self.fovnum = fovnum
        self.offset = None
        self.mask = None
        self.img = None

    def find_intensity_scale(self):
        """
        Returns the value of the first and 99th percentile of intensities in the tile
        """
        return np.percentile(self.img, (1,99.9))

    def scale_intensity(self, p1, p99, convert_to_8bpp=True):
        img = np.clip(((self.img-p1)*(255/(p99-p1))), 0, 255)
        if convert_to_8bpp:
            img = img.astype(np.uint8)
        self.img = img

    def save_as_tiff(self, filename):
        if self.offset is not None:
            d=f"{self.offset[0]} {self.offset[1]} {self.offset[2]}"
        else:
            d=""
        tifffile.imwrite(filename, self.img, description=d)

    def save_as_jp2(self, filename):
        jp2 = glymur.Jp2k(filename)
        jp2[:] = self.img
        # TODO: add offset in metadata
        return

    # def segment(self, checkpoint=None, transpose=False, cached=True):
    #     img, _ = cysgan.run_segmentation(self.img, checkpoint, transpose)
    #     self.img = img

    def from_tiff(self, filename, cached=True):
        img = tifffile.imread(filename)
        self.img = img

    def from_jp2(self, filename, downscale=[1,1,1]):
        # self.img = np.transpose(glymur.Jp2k(filename)[:], (2,1,0))
        self.img = np.transpose(glymur.Jp2k(filename)[:][::downscale[0], ::downscale[1], ::downscale[2]], (2,1,0))

    def from_array(self, array):
        self.img = array

    def from_nd2(self, nd2, downsample=[1,1,1]):
        """
        Loads a tile from a specific FOV

        Args:
            nd2: can be a filename or an existing ND2Reader
            fovnum: the FOV (along the "v" axis) to extract
            downsample: a tuple of 3 items describing a stride to apply in the X Y and Z axis, resulting in a downscale
                        along this axis

        Returns:
            Tile: An initialized tile with image and offset information
        """
        if type(nd2) is str:
            nd2 = ND2Reader(nd2)
        t = Tile()
        ret = t.fast_nd2_read_downscale(nd2, downscale=downsample)
        self.img = ret

    def fast_nd2_read_downscale(self, nd2, downscale):
        """
        Reads the FOV specified by fovnum from the specified nd2 file. This is a fast function that needs a several
        assumptions to be true to work. Notably dtype needs to be int16.
            Arg:
                downscale: [int, int int] steps to apply while reading the image, in Z Y X order
        """
        w = nd2.sizes['x']
        h = nd2.sizes['y']
        sz = w * h
        img = np.zeros((round(nd2.sizes["z"] / downscale[0]),
                             round(nd2.sizes["y"] / downscale[1]),
                             round(nd2.sizes["x"] / downscale[2])), dtype=np.uint16)
        for z in range(0, nd2.sizes['z'], downscale[0]):
            print(f"Z = {z+1}/{nd2.sizes['z']}    ", end="\r")
            image_group_number = nd2._parser._calculate_image_group_number(0, self.fovnum, z)
            chunk = nd2._parser._label_map.get_image_data_location(image_group_number)
            fh = nd2._parser._fh
            fh.seek(chunk)
            # The chunk metadata is always 16 bytes long
            chunk_metadata = fh.read(16)
            header, relative_offset, data_length = struct.unpack("IIQ", chunk_metadata)
            if header != 0xabeceda:
                raise ValueError("The ND2 file seems to be corrupted.")
            # We start at the location of the chunk metadata, skip over the metadata, and then proceed to the
            # start of the actual data field, which is at some arbitrary place after the metadata.
            fh.seek(chunk + 16 + relative_offset)
            data = fh.read(data_length)
            a = np.frombuffer(data[len(data) - sz*2:], dtype=np.uint16)
            a.shape = (w, h)
            img[z//downscale[0],:,:] = a[::downscale[1],::downscale[2]]
        self.img = img

    # def create_hallucinations_mitigation_mask(self):
    #     self.mask=exmfs.cysgan.generate_mask(self.img)

    def apply_mask(self):
        img = self.img.copy()
        img[self.mask == False] = 0
        self.img = img
