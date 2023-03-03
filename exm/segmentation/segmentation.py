import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from cellpose import models,utils,plot,io


def segment_3d(volume,model,downsample=False,chan=0,chan2=0,flow_threshold=0.4,cellprob_threshold=0,do_3D=True):
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
    # DISPLAY RESULTS 3D flows => masks
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