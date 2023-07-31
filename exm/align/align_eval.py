import h5py
import numpy as np
from exm.align.align_utils import alignment_NCC
from exm.utils import configure_logger
from exm.args import Args
from typing import List

logger = configure_logger('ExR-Tools')


def measure_round_alignment_NCC(args: Args, code: int, fov: int) -> List[float]:
    r"""
        Analyzes the alignment between the reference round and another round for a given ROI.
    This function computes the Normalized Cross-Correlation (NCC) between the volumes of the two rounds,
    and returns the distance errors calculated based on the offsets between the two volumes.

    Note: Certain parameters need to be supplied via the Args class:
    - args.nonzero_thresh: Threshold for non-zero pixel count in the volume. 
    - args.N: Number of sub-volumes to test the alignment with.
    - args.subvol_dim: Length of the side of sub-volume, in pixels.
    - args.xystep: Physical pixel size divided by expansion factor, um/voxel in x and y. 
    - args.zstep: Physical z-step size divided by expansion factor, um/voxel in z. 
    - args.pct_thresh: Percentage of pixel intensity values for thresholding.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param code: The code to measure alignment for.
    :type code: int
    :param fov: The FOV to measure alignment for.
    :type fov: int
    :return: List of distance errors after alignment.
    :rtype: List[float]
    """
    distance_errors = []
    logger.info(
        f"Alignment Evaluation: Analyzing alignment between ref round:{args.ref_code} and Code:{code} - FOV:{fov}")

    try:
        with h5py.File(args.h5_path.format(args.ref_code, fov), "r") as f:
            ref_vol = f[args.ref_channel][()]

        with h5py.File(args.h5_path.format(code, fov), "r") as f:
            aligned_vol = f[args.ref_channel][()]

        if np.count_nonzero(aligned_vol) > args.nonzero_thresh:
            ref_vol = (ref_vol - np.min(ref_vol)) / \
                (np.max(ref_vol) - np.min(ref_vol))
            aligned_vol = (aligned_vol - np.min(aligned_vol)) / \
                (np.max(aligned_vol) - np.min(aligned_vol))
            
            keepers = []

            for zz in range(aligned_vol.shape[0]):
                if np.count_nonzero(aligned_vol[zz, :, :]) > 0:
                    keepers.append(zz)

            logger.info(
                f"Alignment Evaluation: Code:{code} - ROI:{fov}, {len(keepers)} slices of {aligned_vol.shape[0]} kept.")

            if len(keepers) < 10:
                logger.info(
                    f"Alignment Evaluation: Code:{code} - FOV:{fov}, fewer than 10 slices. Skipping evaluation...")
            else:
                ref_vol = ref_vol[keepers, :, :]
                aligned_vol = aligned_vol[keepers, :, :]

                distance_errors = alignment_NCC(args, ref_vol, aligned_vol)

        return distance_errors

    except Exception as e:
        logger.error(
            f"Error during NCC alignment measurement for Code: {code}, FOV: {fov}, Error: {e}")
        raise
