import h5py
import numpy as np

from scipy.stats import rankdata

from exm.args import Args
from exm.utils.log import configure_logger
logger = configure_logger('ExSeq-Toolbox')


def quantile_normalization(args: Args, code: int, fov: int) -> None:
    r"""
    Applies quantile normalization to the volumes aligned .h5 file.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param code: The code of the volume to be normalized.
    :type code: int
    :param fov: The field of view of the volume to be normalized.
    :type fov: int
    """
    logger.warn("This function `quantile_normalization` is experimental.")
    try:
        logger.info(
            f"Starting quantile normalization for Code: {code}, FOV: {fov}")

        channels = args.channel_names[:-1]

        with h5py.File(args.h5_path.format(code, fov), "r") as f:
            volumes = [f[channel][()] for channel in channels]

        flattened_volumes = np.concatenate(
            [vol.ravel() for vol in volumes]).reshape(-1, len(channels))

        sorted_volumes = np.sort(flattened_volumes, axis=0)
        mean_volumes = np.mean(sorted_volumes, axis=1)
        rank_volumes = np.empty(flattened_volumes.shape, dtype=int)
        for i in range(flattened_volumes.shape[1]):
            rank_volumes[:, i] = rankdata(
                flattened_volumes[:, i], method='min')

        normalized_volumes = mean_volumes[rank_volumes - 1]

        # Reshape back to original shape
        reshaped_volumes = normalized_volumes.reshape(
            len(channels), *volumes[0].shape)

        # Split into separate volumes
        separate_volumes = np.split(reshaped_volumes, len(channels), axis=0)

        for vol, channel in zip(separate_volumes, channels):
            vol = np.squeeze(vol)
            channel = channel + '_norm'
            with h5py.File(args.h5_path.format(code, fov), "a") as f:
                if channel in f.keys():
                    del f[channel]
                f.create_dataset(channel, vol.shape, dtype=vol.dtype, data=vol)

        logger.info(
            f"Quantile normalization complete for Code: {code}, FOV: {fov}")

    except Exception as e:
        print(f"Error occurred while applying quantile normalization: {e}")
        raise