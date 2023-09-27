
import pickle
import tifffile
import json
import numpy as np
from skimage.measure import label, regionprops
from exm.args import Args

from typing import Tuple, Dict, Union

from exm.utils.utils import configure_logger
logger = configure_logger('ExSeq-Toolbox')


def puncta_assign_gene(args: Args, 
                       fov: int, 
                       option: str = 'original') -> None:
    r"""
    Assign genes to detected puncta based on hamming distance from barcodes.

    This function maps genes to the detected puncta for a given field of view (fov) 
    by comparing the hamming distance from predefined barcodes. It retrieves the 
    barcode mappings for genes, and for each puncta in the given fov, assigns a gene 
    based on the closest hamming distance. 

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param fov: Field of view identifier.
    :type fov: int
    :param option: Determines the operation mode - either 'original' or 'improved'.
                   Determines which puncta results to load and where to save the output.
    :type option: str, default is 'original'

    :return: None. Results are saved as a pickle file.
    """
    from exm.utils.utils import gene_barcode_mapping, retrieve_all_puncta

    def within_hamming_distance(a: str, b: str) -> bool:
        """Check if two strings have a hamming distance less than 2."""
        diff = sum(1 for x, y in zip(a, b) if x != y)
        return diff < 2

    def map_gene_to_puncta(puncta: Dict) -> Dict:
        """Map a gene to a puncta based on barcode hamming distance."""
        for gene, barcode in gene2digit.items():
            if within_hamming_distance(puncta['barcode'], barcode):
                puncta['gene'] = gene
                break
        else:
            puncta['gene'] = 'N/A'
        puncta['fov'] = fov
        return puncta

    df, digit2gene, gene2digit = gene_barcode_mapping(args)

    puncta_list = []

    if option == 'original':
        puncta_results  = retrieve_all_puncta(args, fov)
        output_path = args.puncta_path + f'/fov{fov}/puncta_with_gene.pickle'
    elif option == 'improved':
        with open(args.puncta_path + 'fov{}/improved_puncta_results.pickle'.format(fov), 'rb') as f:
            puncta_results = pickle.load(f)
        output_path = args.puncta_path + f'/fov{fov}/improved_puncta_with_gene.pickle'

    puncta_list = [map_gene_to_puncta(puncta) for puncta in puncta_results]

    with open(output_path, 'wb') as f:
        pickle.dump(puncta_list, f)



def puncta_assign_nuclei(args: Args, 
                         fov: int, 
                         distance_threshold: float = 100, 
                         compare_to_nuclei_surface: bool = True, 
                         num_nearest_nuclei: int = 3, 
                         option: str = 'original') -> None:
    r"""
    Assign puncta to nuclei based on their spatial proximity. For each puncta, the function determines the 
    corresponding nucleus (if any) based on a set distance threshold.

    :param args: Configuration options. This should be an instance of the Args class.
    :type args: Args
    :param fov: Field of view.
    :type fov: int
    :param distance_threshold: Threshold for the maximum allowable distance between puncta and a nucleus.
    :type distance_threshold: float, default is 100
    :param compare_to_nuclei_surface: Flag to determine if distance should be computed to the nuclei surface.
    :type compare_to_nuclei_surface: bool, default is True
    :param num_nearest_nuclei: Number of nearest nuclei to consider when comparing to the nuclei surface.
    :type num_nearest_nuclei: int, default is 3
    :param option: Option to determine which puncta data to use ('original' or 'improved').
    :type option: str, default is 'original'

    :raises: Logs an error message if there's an issue during processing.
    """

    def calculate_distance(coord1: Tuple[float, float, float], 
                           coord2: Tuple[float, float, float]) -> float:
        """Calculate the Euclidean distance between two 3D coordinates."""
        return np.sqrt(np.sum((np.array(coord1) - np.array(coord2)) ** 2))

    def min_distance_to_object(point: Tuple[float, float, float], 
                               object_coords: np.ndarray) -> float:
        """Calculate the minimum distance from a point to any point in a 3D object."""
        distances = np.linalg.norm(object_coords - point, axis=1)
        return np.min(distances)

    def get_nuclei_centroid(mask: np.ndarray) -> Dict[int, Tuple[float, float, float]]:
        """Extract centroid coordinates and associate them with their labels."""
        nuclei_centroid_dict = {}
        props = regionprops(mask)
        for prop in props:
            centroid = prop.centroid
            label = prop.label
            nuclei_centroid_dict[label] = centroid
        return nuclei_centroid_dict

    count_within = 0
    count_nearby = 0
    count_gene = 0
    logger.info(f"Processing puncta_assign_nuclei FOV: {fov}")

    try:
        if option == 'original':
            with open(args.puncta_path + f"fov{fov}/puncta_with_gene.pickle", "rb") as f:
                    puncta_with_genes = pickle.load(f)
        elif option == 'improved':
                with open(args.puncta_path + f"fov{fov}/improved_puncta_with_gene.pickle", "rb") as f:
                    puncta_with_genes = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load puncta data: {e}")
        return

    try:
        nuclei_mask = tifffile.imread(args.puncta_path + f"/nuclei_segmentation/fov{fov}_mask.tif")

    except Exception as e:
        logger.error(f"Failed to load nuclei mask: {e}")
        return

    nuclei_centroids = get_nuclei_centroid(nuclei_mask)

    for i, puncta in enumerate(puncta_with_genes):
        if puncta.get("gene") != "N/A":
            count_gene += 1
            try:
                if nuclei_mask[tuple(puncta.get("position"))] != 0:
                    puncta['nuclei'] = nuclei_mask[tuple(
                        puncta.get("position"))]
                    count_within += 1
            except Exception as e:
                logger.error(f"Error processing puncta {i}: {e}")

    for i, puncta in enumerate(puncta_with_genes):
        if puncta.get("gene") != "N/A" and not puncta.get('nuclei'):
            distance_label_list = [(calculate_distance(puncta.get(
                "position"), centroid), label) for label, centroid in nuclei_centroids.items()]

            # measure the closest distance between the puncta and the nuclei with closest centroid distance to the puncta "num_nearest nuclei" determine how many nuclei consider while comparing
            if compare_to_nuclei_surface:
                distance_label_list.sort(key=lambda x: x[0])
                nearest_nuclei = [
                    label for _, label in distance_label_list[:num_nearest_nuclei]]
                final_distances = []

                for nuclei_label in nearest_nuclei:
                    object_coords = np.column_stack(
                        np.where(nuclei_mask == nuclei_label))
                    min_distance = min_distance_to_object(
                        puncta.get("position"), object_coords)
                    final_distances.append((min_distance, nuclei_label))

                final_distances.sort(key=lambda x: x[0])
                if final_distances[0][0] < distance_threshold:
                    puncta['nuclei'] = final_distances[0][1]
                    count_nearby += 1
            else:
                distance_label_list.sort(key=lambda x: x[0])
                if distance_label_list[0][0] < distance_threshold:
                    puncta['nuclei'] = distance_label_list[0][1]
                    count_nearby += 1


    if option == 'original':
        output_path = args.puncta_path + \
            f"fov{fov}/puncta_with_nuclei.pkl"
    elif option == 'improved':
        output_path = args.puncta_path + \
            f"fov{fov}/improved_puncta_with_nuclei.pkl"

    try:
        with open(output_path, 'wb') as pickle_file:
            pickle.dump(puncta_with_genes, pickle_file)

        logger.info(
            f"Done puncta_assign_nuclei FOV: {fov}, puncta within nuclei {count_within}, puncta nearby nuclei {count_nearby} , total puncta with gene correspondence {count_gene}")

    except Exception as e:
        logger.error(f"Failed to save output: {e}")
