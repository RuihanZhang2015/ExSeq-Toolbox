
import pickle


def assign_gene_fov(args, fov, option = 'original'):
    r"""
    This function assigns genes to the detected puncta for a specified field of view (fov) based on hamming distance from barcodes. The function first retrieves the barcode mappings for genes, then iterates over each puncta in the given fov, and assigns a gene to the puncta based on the closest hamming distance. The function can operate in two modes, 'original' and 'improved', determined by the 'option' parameter. In both cases, the function saves the updated puncta list with assigned genes to a pickle file.

    :param args: Configuration options.
    :type args.Args: args.Args instance
    :param fov: Field of view.
    :type fov: int
    :param option: Operation mode, either 'original' or 'improve'. If 'original', the function retrieves the `conslidation_codes` results and saves output to 'puncta_with_gene.pickle'. If 'improve', the function loads results from 'improved_puncta_results.pickle' and saves output to 'improved_puncta_with_gene.pickle'. Default is 'original'.
    :type option: str, optional

    :returns: No return value. The function saves the puncta list with assigned genes to a pickle file in a subdirectory named 'fov{fov}' under the directory specified by args.puncta_path.
    """

    from exm.utils import gene_barcode_mapping,retrieve_all_puncta

    def within_hamming_distance(a,b):
        diff = 0
        for x,y in zip(a,b):
            if x!=y:
                diff +=1
            if diff>=2:
                return False
        return True

    def map_gene(puncta):
        for gene in gene2digit.keys():
            if within_hamming_distance(puncta['barcode'],gene2digit[gene]):
                return {
                    **puncta,
                    'fov':fov,
                    'gene':gene
                }
        return {
                 **puncta,
                'fov':fov,
                'gene':'N/A'
        }
        
    df,digit2gene,gene2digit = gene_barcode_mapping(args) 

    puncta_list = []

    if option == 'original':
        result = retrieve_all_puncta(args,fov)
    elif option == 'improve':
        with open(args.puncta_path + 'fov{}/improved_puncta_results.pickle'.format(fov),'rb') as f:
            result = pickle.load(f)
  
    for puncta in result:
        new_puncta = map_gene(puncta)
        puncta_list.append(new_puncta)
        
    if option == 'original':
        with open(args.puncta_path + 'fov{}/puncta_with_gene.pickle'.format(fov), 'wb') as f:
            pickle.dump(puncta_list,f)
    elif option == 'improve':
        with open(args.puncta_path + 'fov{}/improved_puncta_with_gene.pickle'.format(fov), 'wb') as f:
            pickle.dump(puncta_list,f)
