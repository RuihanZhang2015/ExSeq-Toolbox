import os
import pickle
import plotly.subplots as sp
import plotly.graph_objs as go


def puncta_analysis(args, fov, improved=False):
    r"""Analyzes puncta within a specified field of view (fov).
    This function opens a pickle file containing consolidated puncta data, and performs an analysis to collect statistics on these puncta. The statistics include total puncta, puncta with and without reference codes, reference code counts, puncta with and without missing codes, missing code counts, unique reference codes, puncta without missing codes and gene, etc.
    The results are stored in a dictionary and written to a pickle file with a specific file name based on whether the puncta data was improved or not.

    :param args.Args args: Configuration options.
    :param int fov: Field of view.
    :param bool improved: If True, the function retrieves data from 'improved_puncta_with_gene.pickle' file, else retrieves data from 'puncta_with_gene.pickle' file. Default is False.

    :returns: No return value. The function saves the analysis result to a pickle file named 'improved_puncta_analysis.pickle' or 'original_puncta_analysis.pickle' in a subdirectory of each fov, depending on the `improved` parameter.
    """
    if improved:
        with open(args.puncta_path + '/fov{}/improved_puncta_with_gene.pickle'.format(fov), 'rb') as f:
            consolidated_puncta = pickle.load(f)
    else:
        with open(args.puncta_path + '/fov{}/puncta_with_gene.pickle'.format(fov), 'rb') as f:
            consolidated_puncta = pickle.load(f)

    total_puncta = len(consolidated_puncta)
    puncta_without_ref_code = 0
    puncta_with_ref_code = 0
    ref_code_counts = dict()
    total_ref_code_counts = 0
    puncta_with_missing_codes = 0
    puncta_without_missing_codes = 0
    missing_code_counts = dict()
    missing_codes_distribution = dict()
    unique_ref_codes = set()
    ref_code_frequency = dict()

    puncta_without_missing_codes_and_gene = 0
    puncta_without_missing_codes_and_no_gene = 0
    gene_distribution = dict()

    for puncta in consolidated_puncta:
        has_ref_code = False
        code_count = 0
        for key, value in puncta.items():
            if key.startswith('code'):
                code_count += 1
                if 'ref_code' in value:
                    ref_code_value = value['ref_code']
                    if 0 <= ref_code_value <= 7:
                        has_ref_code = True
                        total_ref_code_counts += 1
                        unique_ref_codes.add(ref_code_value)
                        ref_code_frequency[ref_code_value] = ref_code_frequency.get(
                            ref_code_value, 0) + 1
                        ref_code_counts[key] = ref_code_counts.get(key, 0) + 1
        if has_ref_code:
            puncta_with_ref_code += 1
        else:
            puncta_without_ref_code += 1
        if code_count < 7:
            missing_codes = 7 - code_count
            missing_codes_distribution[missing_codes] = missing_codes_distribution.get(
                missing_codes, 0) + 1
            puncta_with_missing_codes += 1
            for i in range(7):
                code_key = 'code' + str(i)
                if code_key not in puncta:
                    missing_code_counts[code_key] = missing_code_counts.get(
                        code_key, 0) + 1
        else:
            puncta_without_missing_codes += 1
            if puncta['gene'] == 'N/A':
                puncta_without_missing_codes_and_no_gene += 1
            else:
                puncta_without_missing_codes_and_gene += 1
                gene_distribution[puncta['gene']] = gene_distribution.get(
                    puncta['gene'], 0) + 1

    result = {
        'total_puncta': total_puncta,
        'puncta_without_ref_code': puncta_without_ref_code,
        'puncta_with_ref_code': puncta_with_ref_code,
        'total_ref_code_counts': total_ref_code_counts,
        'ref_code_counts': ref_code_counts,
        'puncta_with_missing_codes': puncta_with_missing_codes,
        'puncta_without_missing_codes': puncta_without_missing_codes,
        'missing_code_counts': missing_code_counts,
        'missing_codes_distribution': missing_codes_distribution,
        'unique_ref_codes': unique_ref_codes,
        'num_unique_ref_codes': len(unique_ref_codes),
        'ref_code_frequency': ref_code_frequency,
        'puncta_without_missing_codes_and_gene': puncta_without_missing_codes_and_gene,
        'puncta_without_missing_codes_and_no_gene': puncta_without_missing_codes_and_no_gene,
        'gene_distribution': gene_distribution
    }

    if improved:
        file_name = 'fov{}/improved_puncta_analysis.pickle'.format(fov)
    else:
        file_name = 'fov{}/original_puncta_analysis.pickle'.format(fov)

    with open(args.puncta_path + file_name, 'wb') as f:
        pickle.dump(result, f)


def aggregate_puncta_analysis(args, improved=False):
    r"""Aggregates puncta analysis results across all fields of view (fovs).
    This function opens pickle files containing the results of puncta analysis for each fov specified in the args parameter. It then aggregates these results, summing or unionizing values as appropriate based on the data type.
    The aggregated results are written to a pickle file with a specific name based on whether the original puncta data was improved or not.

    :param args.Args args: Configuration options. The `fovs` attribute should contain a list of fovs to include in the aggregation.
    :param bool improved: If True, the function retrieves data from 'improved_puncta_analysis.pickle' files, else retrieves data from 'original_puncta_analysis.pickle' files. Default is False.

    :returns: No return value. The function saves the aggregated analysis result to a pickle file named 'aggregate_improved_puncta_analysis.pickle' or 'aggregate_orginal_puncta_analysis.pickle' in the directory specified by args.puncta_path, depending on the `improved` parameter.
    """
    aggregate_result = dict()

    for fov in args.fovs:
        if improved:
            file_name = 'fov{}/improved_puncta_analysis.pickle'.format(fov)
        else:
            file_name = 'fov{}/original_puncta_analysis.pickle'.format(fov)

        with open(args.puncta_path + file_name, 'rb') as f:
            result = pickle.load(f)

        if not aggregate_result:
            aggregate_result = {key: value.copy() if isinstance(
                value, dict) else value for key, value in result.items()}

        else:
            for key, value in result.items():
                if isinstance(value, dict):
                    for inner_key, inner_value in value.items():
                        # Check if the key exists in the aggregate_result dictionary
                        if inner_key in aggregate_result[key]:
                            aggregate_result[key][inner_key] += inner_value
                        else:
                            # If the key doesn't exist, create it with a default value
                            aggregate_result[key][inner_key] = inner_value
                elif isinstance(value, int) or isinstance(value, float):
                    aggregate_result[key] += value
                elif isinstance(value, set):
                    aggregate_result[key] = aggregate_result[key].union(value)

    aggregate_result['num_unique_ref_codes'] = len(
        aggregate_result['unique_ref_codes'])

    if improved:
        with open(args.puncta_path + '/aggregate_improved_puncta_analysis.pickle', 'wb') as f:
            pickle.dump(aggregate_result, f)
    else:
        with open(args.puncta_path + '/aggregate_orginal_puncta_analysis.pickle', 'wb') as f:
            pickle.dump(aggregate_result, f)


def plot_benchmark_data(args, mode='single', fov=None):
    r"""Generates a comparative dashboard for puncta analysis results in a single field of view (fov) or across all fovs.
    This function opens pickle files containing the original and improved results of puncta analysis for a specified fov or for all fovs. The results are used to generate various bar charts showing comparisons between original and improved puncta counts, distributions, and other metrics. These charts are saved as a HTML file.

    :param args.Args args: Configuration options. 
    :param str mode: If 'single', the function retrieves data for a specific fov. If 'all', the function retrieves aggregate data for all fovs. Default is 'single'.
    :param int fov: The specific fov to retrieve data for if mode is 'single'. Default is None.

    :returns: No return value. The function saves the generated dashboard to a HTML file named 'puncta_comparison_dashboard.html' or 'overall_puncta_comparison_dashboard.html' in the directory specified by args.puncta_path, depending on the `mode` parameter.
    """
    if mode == 'single':
        file_name = 'fov{}/puncta_comparison_dashboard.html'.format(fov)
        with open(args.puncta_path + 'fov{}/original_puncta_analysis.pickle'.format(fov), 'rb') as f:
            puncta_original = pickle.load(f)

        with open(args.puncta_path + 'fov{}/improved_puncta_analysis.pickle'.format(fov), 'rb') as f:
            puncta_improved = pickle.load(f)

    if mode == 'all':
        fov = 'All'
        file_name = 'overall_puncta_comparison_dashboard.html'
        with open(args.puncta_path + 'aggregate_orginal_puncta_analysis.pickle', 'rb') as f:
            puncta_original = pickle.load(f)

        with open(args.puncta_path + 'aggregate_improved_puncta_analysis.pickle', 'rb') as f:
            puncta_improved = pickle.load(f)

    # Sort keys in missing_code_counts dictionaries
    missing_code_counts_original = {k: v for k, v in sorted(
        puncta_original['missing_code_counts'].items(), key=lambda item: int(item[0][4:]))}
    missing_code_counts_improved = {k: v for k, v in sorted(
        puncta_improved['missing_code_counts'].items(), key=lambda item: int(item[0][4:]))}

    # Sort keys in ref_code_counts dictionaries
    ref_code_counts_original = {k: v for k, v in sorted(
        puncta_original['ref_code_counts'].items(), key=lambda item: int(item[0][4:]))}
    ref_code_counts_improved = {k: v for k, v in sorted(
        puncta_improved['ref_code_counts'].items(), key=lambda item: int(item[0][4:]))}

    # Sort keys in gene_distribution dictionaries
    gene_distribution_original = {k: v for k, v in sorted(
        puncta_original['gene_distribution'].items(), key=lambda item: item[1], reverse=True)}
    gene_distribution_improved = {k: v for k, v in sorted(
        puncta_improved['gene_distribution'].items(), key=lambda item: item[1], reverse=True)}

    # Create subplot
    fig = sp.make_subplots(
        rows=4, cols=2,
        subplot_titles=['Puncta Counts',
                        'Distribution of Imporved Puncta',
                        'Distribution of Missing Rounds',
                        'Distribution of Reference Round (Improved)',
                        'Complete Barcode Puncta (with/without Gene)',
                        'Gene Distribution',
                        'Puncta vs Number of Missing Codes'])

    fig.add_trace(go.Bar(name='Puncta Count (Original)',
                         x=['Total Puncta', 'Puncta complete barcode',
                             'Puncta missing barcode'],
                         y=[puncta_original['total_puncta'], puncta_original['puncta_without_missing_codes'],
                             puncta_original['puncta_with_missing_codes']],
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=1, col=1)

    fig.add_trace(go.Bar(name='Puncta Count (Improved)',
                         x=['Total Puncta', 'Puncta complete barcode',
                             'Puncta missing barcode', 'Improved Puncta', 'Non-Improved Puncta'],
                         y=[puncta_improved['total_puncta'], puncta_improved['puncta_without_missing_codes'],
                            puncta_improved['puncta_with_missing_codes'], puncta_improved['puncta_with_ref_code'],
                            puncta_improved['puncta_without_ref_code'] - puncta_original['puncta_without_missing_codes']],
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=1, col=1)

    fig.add_trace(go.Bar(name='Puncta Distribution (Improved)',
                         x=list(ref_code_counts_improved.keys()),
                         y=list(ref_code_counts_improved.values()),
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=1, col=2)

    # Subplot for entities with missing codes
    fig.add_trace(go.Bar(name='Missing Puncta Distribution (Original)',
                         x=list(missing_code_counts_original.keys()),
                         y=list(missing_code_counts_original.values()),
                         hovertemplate='Count: %{y}<extra></extra>'), row=2, col=1)

    fig.add_trace(go.Bar(name='Missing Puncta Distribution (Improved)',
                         x=list(missing_code_counts_improved.keys()),
                         y=list(missing_code_counts_improved.values()),
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=2, col=1)

    fig.add_trace(go.Bar(name='Reference Round Distribution',
                         x=list(puncta_improved['ref_code_frequency'].keys()),
                         y=list(
                             puncta_improved['ref_code_frequency'].values()),
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=2, col=2)
    fig.update_xaxes(tickmode='array', tickvals=list(
        puncta_improved['ref_code_frequency'].keys()), row=2, col=2)

    # Subplot for puncta without missing codes
    fig.add_trace(go.Bar(name='Complete Barcode Puncta (Original)',
                         x=['With gene', 'Without gene'],
                         y=[puncta_original['puncta_without_missing_codes_and_gene'],
                             puncta_original['puncta_without_missing_codes_and_no_gene']],
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=3, col=1)

    fig.add_trace(go.Bar(name='Complete Barcode Puncta (Improved)',
                         x=['With gene', 'Without gene'],
                         y=[puncta_improved['puncta_without_missing_codes_and_gene'],
                             puncta_improved['puncta_without_missing_codes_and_no_gene']],
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=3, col=1)

    # Subplot for gene_distribution
    fig.add_trace(go.Bar(name='Gene Distribution (Original)',
                         x=list(gene_distribution_original.keys()),
                         y=list(gene_distribution_original.values()),
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=3, col=2)

    fig.add_trace(go.Bar(name='Gene Distribution (Improved)',
                         x=list(gene_distribution_improved.keys()),
                         y=list(gene_distribution_improved.values()),
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=3, col=2)

    fig.add_trace(go.Bar(name='Missing Codes Distribution (Original)',
                         x=list(
                             puncta_original['missing_codes_distribution'].keys()),
                         y=list(
                             puncta_original['missing_codes_distribution'].values()),
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=4, col=1)

    fig.add_trace(go.Bar(name='Missing Codes Distribution (Improved)',
                         x=list(
                             puncta_improved['missing_codes_distribution'].keys()),
                         y=list(
                             puncta_improved['missing_codes_distribution'].values()),
                         hovertemplate='Count: %{y}<extra></extra>'),
                  row=4, col=1)

    fig.update_xaxes(tickmode='array', tickvals=list(
        puncta_original['missing_codes_distribution'].keys()), row=4, col=1)

    fig.update_layout(height=1900, width=1500, barmode='group',
                      title_text="Puncta Extraction Summary (FOV {})".format(fov), font=dict(size=18))

    fig.write_html(os.path.join(args.puncta_path, file_name))
    fig.show()
