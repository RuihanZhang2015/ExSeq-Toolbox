import numpy as np
from skimage.filters import threshold_otsu
from scipy.fftpack import fftn, ifftn
from exm.args import Args
from typing import List, Dict

from exm.utils import configure_logger

logger = configure_logger('ExR-Tools')

def template_matching(T: np.ndarray, I: np.ndarray, IdataIn: Dict = None) -> np.ndarray:
    """
    Implements template matching between two images using Fourier transform.

    :param T: The template image.
    :type T: np.ndarray
    :param I: The image to be matched.
    :type I: np.ndarray
    :param IdataIn: A dictionary containing additional image data (optional).
    :type IdataIn: Dict, optional
    :return: The Normalized Cross-Correlation (NCC) between the template and the image.
    :rtype: np.ndarray
    """
    def unpadarray(A: np.ndarray, Bsize: np.ndarray) -> np.ndarray:
        Bsize = np.array(Bsize)
        Bstart = np.ceil((np.array(A.shape) - Bsize) / 2).astype(int)
        Bend = Bstart + Bsize
        if len(A.shape) == 2:
            B = A[Bstart[0]:Bend[0], Bstart[1]:Bend[1]]
        elif len(A.shape) == 3:
            B = A[Bstart[0]:Bend[0], Bstart[1]:Bend[1], Bstart[2]:Bend[2]]
        return B

    def local_sum(I: np.ndarray, T_size: np.ndarray) -> np.ndarray:
        T_size = np.array(T_size)
        # Add padding to the image
        B = np.pad(I, ((T_size[0], T_size[0]), (T_size[1], T_size[1]), (T_size[2], T_size[2])), mode='constant')

        s = np.cumsum(B, axis=0)
        c = s[T_size[0]:-1, :, :] - s[:-T_size[0]-1, :, :]
        s = np.cumsum(c, axis=1)
        c = s[:, T_size[1]:-1, :] - s[:, :-T_size[1]-1, :]
        s = np.cumsum(c, axis=2)
        local_sum_I = s[:, :, T_size[2]:-1] - s[:, :, :-T_size[2]-1]

        return local_sum_I

    try:
        # Convert images to double
        T = T.astype(float)
        I = I.astype(float)

        T_size = np.array(T.shape)
        I_size = np.array(I.shape)
        outsize = I_size + T_size - 1
        Idata = {}

        # calculate correlation in frequency domain
        FT = fftn(np.flip(T, axis=(0, 1, 2)), outsize)
        FI = fftn(I, outsize)
        Icorr = np.real(ifftn(FI * FT))

        # Calculate Local Quadratic sum of Image and Template
        if IdataIn is None or 'LocalQSumI' not in IdataIn:
            Idata['LocalQSumI'] = local_sum(I * I, T_size)
        else:
            Idata['LocalQSumI'] = IdataIn['LocalQSumI']

        QSumT = np.sum(T**2)

        # SSD between template and image
        I_SSD = Idata['LocalQSumI'] + QSumT - 2 * Icorr

        # Normalize to range 0..1
        I_SSD = I_SSD - np.min(I_SSD)
        I_SSD = 1 - I_SSD / np.max(I_SSD)

        # Remove padding
        I_SSD = unpadarray(I_SSD, I_size)

        if len(Idata) > 0:
            # Normalized cross correlation STD
            if IdataIn is None or 'LocalSumI' not in IdataIn:
                Idata['LocalSumI'] = local_sum(I, T_size)
            else:
                Idata['LocalSumI'] = IdataIn['LocalSumI']

            # Standard deviation
            if IdataIn is None or 'stdI' not in IdataIn:
                Idata['stdI'] = np.sqrt(np.maximum(Idata['LocalQSumI'] - (Idata['LocalSumI']**2) / np.prod(T_size), 0))
            else:
                Idata['stdI'] = IdataIn
            stdT = np.sqrt(T.size - 1) * np.std(T, ddof=1)

            # Mean compensation
            meanIT = Idata['LocalSumI'] * np.sum(T) / np.prod(T_size)
            I_NCC = 0.5 + (Icorr - meanIT) / (2 * stdT * np.maximum(Idata['stdI'], stdT / 100000))

            # Remove padding
            I_NCC = unpadarray(I_NCC, I_size)

        return I_NCC

    except Exception as e:
        logger.error(f"Error during template matching, Error: {e}")
        raise


def alignment_NCC(args: Args, vol1: np.ndarray, vol2: np.ndarray) -> List[float]:
    r"""
    Measures the alignment of two images using Normalized Cross-Correlation (NCC). expected shape `[Z,Y,X]`

    :param config: Configuration options.
    :type config: Config
    :param vol1: The first volume (reference volume) for alignment comparison.
    :type vol1: np.ndarray
    :param vol2: The second volume (aligned volume) for alignment comparison.
    :type vol2: np.ndarray
    :return: List of distance errors after alignment.
    :rtype: List[float]
    """
    xy_vol_half = int(args.subvol_dim/2)
    z_vol_half = int(min(np.floor(vol1.shape[0]/2)-1,np.floor(xy_vol_half*(args.xystep/args.zstep))))

    offsets_total = np.full((args.N, 3), -30)

    # Calculate the percentile of the values in vol1 and vol2 specified by config.pct_thresh
    thresh_vol1 = np.percentile(vol1, args.pct_thresh)
    thresh_vol2 = np.percentile(vol2, args.pct_thresh)

    xpos = np.random.randint(0, vol1.shape[2], args.N)
    ypos = np.random.randint(0, vol1.shape[1], args.N)
    zpos = np.random.randint(0, vol1.shape[0], args.N)

    i = 0
    while i < args.N:
        try:
            # Generate new random positions for xpos and ypos
            xpos[i] = np.random.randint(0, vol1.shape[2], 1)[0]
            ypos[i] = np.random.randint(0, vol1.shape[1], 1)[0]

            # If the first dimension of vol1 is less than 50, set zpos[i] to the middle
            # Otherwise, generate a new random position for zpos
            if vol1.shape[0] < 50:
                zpos[i] = vol1.shape[0] // 2
            else:
                zpos[i] = np.random.randint(0, vol1.shape[0],1)[0]

            # Check that the random position is within bounds
            if not (xpos[i] > xy_vol_half and xpos[i] < vol1.shape[2] - xy_vol_half):
                continue
            elif not (ypos[i] > xy_vol_half and ypos[i] < vol1.shape[1] - xy_vol_half):
                continue
            elif not (zpos[i] > z_vol_half and zpos[i] < vol1.shape[0] - z_vol_half):
                continue

            # Create the subvolumes 
            subvolume1 = vol1[zpos[i]-z_vol_half:zpos[i]+z_vol_half+1,
                            xpos[i]-xy_vol_half:xpos[i]+xy_vol_half+1,
                            ypos[i]-xy_vol_half:ypos[i]+xy_vol_half+1
                            ]
            
            subvolume2 = vol2[zpos[i]-z_vol_half:zpos[i]+z_vol_half+1,
                            xpos[i]-xy_vol_half:xpos[i]+xy_vol_half+1,
                            ypos[i]-xy_vol_half:ypos[i]+xy_vol_half+1
                            ]

            # Flatten the subvolumes
            subvec1 = subvolume1.flatten()
            subvec2 = subvolume2.flatten()
            # Binarize the subvolumes using Otsu's method
            thresh_vol1 = threshold_otsu(subvec1)
            thresh_vol2 = threshold_otsu(subvec2)
            subvec1_bin = subvec1 > thresh_vol1
            subvec2_bin = subvec2 > thresh_vol2

            # Check if the subvolumes contain enough non-zero pixels
            if np.mean(subvec1_bin) <= 0.01 or np.mean(subvec2_bin) <= 0.01:
                continue

            I_NCC = template_matching(subvolume1,subvolume2)
            idx = np.unravel_index(np.argmax(I_NCC, axis=None), I_NCC.shape)      
            offsets_total[i, :] = np.array(idx) - np.ceil(np.array(I_NCC.shape) / 2)
            
            i += 1

        except Exception as e:
            logger.error(
                f"Error during NCC alignment calculation, Error: {e}")
            raise

    distance_errors = np.sqrt((args.xystep * offsets_total[:, 0]) ** 2 + (args.xystep * offsets_total[:, 1]) ** 2 + (args.zstep * offsets_total[:, 2]) ** 2)
  
    return distance_errors