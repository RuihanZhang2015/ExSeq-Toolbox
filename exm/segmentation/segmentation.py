import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from cellpose import models,utils,plot,io


def segment_3d(volume,model,downsample=False,chan=0,chan2=0,flow_threshold=0.4,cellprob_threshold=0,do_3D=True):
    r"""
    Performs 3D segmentation on the given volume using the provided model.

    :param volume: The volume to segment.
    :type volume: numpy.ndarray
    :param model: The model to use for segmentation.
    :type model: cellpose.models.CellposeModel
    :param downsample: Whether to downsample the volume before segmentation. Default is False.
    :type downsample: bool
    :param chan: The channel to use for segmentation. Default is 0.
    :type chan: int
    :param chan2: The second channel to use for segmentation. Default is 0.
    :type chan2: int
    :param flow_threshold: The flow threshold for the segmentation. Default is 0.4.
    :type flow_threshold: float
    :param cellprob_threshold: The cell probability threshold for the segmentation. Default is 0.
    :type cellprob_threshold: float
    :param do_3D: Whether to perform 3D segmentation. Default is True.
    :type do_3D: bool
    :return: The segmented masks.
    :rtype: numpy.ndarray
    """
    volume = np.expand_dims(volume, axis=1)
    if downsample:
        volume = ndimage.zoom(volume, (1,1,0.25,0.25), order= 1)    
    masks, flows, styles = model.eval(volume, 
                                  channels=[chan, chan2],
                                  diameter=model.diam_labels,
                                  flow_threshold=flow_threshold,
                                  cellprob_threshold=cellprob_threshold,
                                  do_3D=do_3D
                                  )
    if downsample:
        masks = ndimage.zoom(masks,(1,4,4), order= 0)
    return masks


def display_3d_masks(image,masks):
    r"""
    Displays the 3D masks on the given image.

    :param image: The image to display.
    :type image: numpy.ndarray
    :param masks: The masks to overlay on the image.
    :type masks: numpy.ndarray
    """
    plt.figure(figsize=(15,30))
    for i,iplane in enumerate(np.arange(0,image.shape[0],50,int)):
        img0 = plot.image_to_rgb(image[iplane, [0,0]].copy(), channels=[0,0])
        plt.subplot(6,3,i+1)
        outlines = utils.masks_to_outlines(masks[iplane])
        outX, outY = np.nonzero(outlines)
        imgout= img0.copy()
        imgout[outX, outY] = np.array([255,75,75])
        plt.imshow(imgout)
        plt.title('Z-plane = %d'%iplane)