"""
Alignment evaluation module is dedicated to assessing the quality of alignment between different rounds of volumetric microscopy images within a field of view (FOV). It leverages Normalized Cross-Correlation (NCC) to quantify alignment, offering a robust measure of similarity between the reference round and other rounds.
"""
import os
import json
import h5py
import numpy as np
from exm.align.align_utils import alignment_NCC
from exm.utils import configure_logger
from exm.args import Args
from typing import List,Dict,Optional

logger = configure_logger('ExR-Tools')


def measure_round_alignment_NCC(args: Args, code: int, fov: int) -> List[float]:
    r"""
    Analyzes the alignment between the reference round and another round for a given ROI.
    
    This function computes the Normalized Cross-Correlation (NCC) between the volumes of the two rounds,
    and returns the distance errors calculated based on the offsets between the two volumes.

    **Note:**
    Certain parameters are needed to be supplied via the Args class:

        - `args.nonzero_thresh`: Threshold for non-zero pixel count in the volume. 
        - `args.N` : Number of sub-volumes to test the alignment with.
        - `args.subvol_dim` : Length of the side of sub-volume, in pixels.
        - `args.xystep`: Physical pixel size divided by expansion factor, um/voxel in x and y. 
        - `args.zstep` : Physical z-step size divided by expansion factor, um/voxel in z. 
        - `args.pct_thresh`: Percentage of pixel intensity values for thresholding.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param code: The code to measure alignment NCC for.
    :type code: int
    :param fov: The FOV to measure alignment NCC for.
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

                eval_file = os.path.join(
                    args.processed_data_path, "alignment_evaluation", f"FOV{fov}", f"NCC_FOV{fov}.json")

                eval_data = {}
                if os.path.exists(eval_file):
                    with open(eval_file, 'r') as f:
                        eval_data = json.load(f)

                eval_data[f"R{args.ref_code}-{code}"] = distance_errors.tolist()

                with open(eval_file, 'w') as f:
                    json.dump(eval_data, f)

        return distance_errors

    except Exception as e:
        logger.error(
            f"Error during NCC alignment measurement for Code: {code}, FOV: {fov}, Error: {e}")
        raise


def plot_alignment_evaluation(args: Args, fov: int, percentile: int = 95, save_fig: Optional[bool] = False) -> None:
    
    """
    Plots alignment evaluation data using a violin plot.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param fov: The fov to plot.
    :type fov: int
    :param percentile: The percentile to filter the data. Default is 95.
    :type percentile: int
    :param save_fig: Flag to save the figure. If False, the figure is not saved. Default is False.
    :type save_fig: Optional[bool]

    :raises: Raises an exception if an error occurs during the plotting process.

    :note: The function shows the plot using plt.show() and optionally saves it to `<processed_data_path>/alignment_evaluation/ROI<roi>/Alignment_Evaluation_ROI<roi>.json` if save_dir is True.
    """
    
    def load_eval_data(args: Args, fov: int) -> dict:
        eval_file = os.path.join(args.processed_data_path, "alignment_evaluation", f"FOV{fov}", f"NCC_fov{fov}.json")
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        return eval_data
           
    import seaborn as sns
    import matplotlib.pyplot as plt

    try:
        eval_data = load_eval_data(args, fov)
        
        sorted_keys = sorted(eval_data.keys(), key=lambda x: int(x.split('-')[1]))

        keys_list = []
        arrays_list = []

        for key in sorted_keys:
            value = eval_data[key]
            keys_list.append(key)
            eval_array = np.array(value)
            percentile_value = np.percentile(eval_array, percentile)
            eval_filtered = eval_array[eval_array <= percentile_value]
            arrays_list.append(np.array(eval_filtered))

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=arrays_list, inner="quartile")
        plt.ylabel("Registration Error (\u00B5m)")
        plt.xticks(np.arange(len(keys_list)), keys_list, rotation=45)
        plt.title(f"Violin Plot of Alignment Evaluation Results FOV{fov}")

        if save_fig:
            save_path = os.path.join(args.processed_data_path,"alignment_evaluation", f"FOV{fov}",f"Alignment_Evaluation_FOV{fov}.png")
            plt.savefig(save_path)
            logger.info(f'Figure saved at {save_path}')

        plt.show()

        logger.info(f'Successfully plotted alignment evaluation for FOV: {fov}.')

    except Exception as e:
        logger.error(f'Error during alignment evaluation plotting for FOV: {fov}, Error: {e}')
        raise



def calculate_alignment_evaluation_ci(args: Args, fov: int,  
                                      ci: float = 95, 
                                      percentile_filter: float = 95) -> Dict[str, Dict[str, float]]:
    r"""
    Calculate the confidence interval (CI) for alignment evaluation and save it as a JSON file.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param fov: The FOV for which alignment evaluation will be calculated.
    :type fov: int
    :param ci: Confidence level for CI calculation. Default is 95.
    :type ci: float
    :param percentile_filter: Percentile value for filtering the data before CI calculation. Default is 95.
    :type percentile_filter: float
    :return: A dictionary containing the lower and upper bounds of CI for each alignment key.
    :rtype: Dict[str, Dict[str, float]]
    """
    
    def bootstrap_mean_ci(data: np.array, n_bootstrap_samples: int, ci: float) -> (float, float):
        bootstrap_means = np.zeros(n_bootstrap_samples)
        for i in range(n_bootstrap_samples):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means[i] = np.mean(bootstrap_sample)
        lower = (100 - ci) / 2
        upper = 100 - lower
        lower_bound, upper_bound = np.percentile(bootstrap_means, [lower, upper])
        return lower_bound, upper_bound

    ci_data = {}
    eval_file_path = os.path.join(args.processed_data_path, "alignment_evaluation", f"ROI{roi}", f"NCC_ROI{roi}.json")
    
    try:
        with open(eval_file_path, 'r') as f:
            eval_data = json.load(f)
        
        sorted_keys = sorted(eval_data.keys(), key=lambda x: int(x.split('-')[1]))

        for key in sorted_keys:
            value = eval_data[key]
            eval_array = np.array(value)
            percentile_value = np.percentile(eval_array, percentile_filter)
            eval_filtered = eval_array[eval_array <= percentile_value]
            
            lower_bound, upper_bound = bootstrap_mean_ci(eval_filtered, len(eval_data)//10, ci)
            ci_data[key] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
        
        ci_file_path = os.path.join(args.processed_data_path, "alignment_evaluation", f"FOV{fov}", f"CI_FOV{fov}.json")
        with open(ci_file_path, 'w') as f:
            json.dump(ci_data, f)
        
        logger.info(f"CI calculation for FOV:{fov} completed successfully.")
        
    except Exception as e:
        logger.error(f"Error during CI calculation for FOV:{fov}. Error: {e}")
        raise
    
    return ci_data