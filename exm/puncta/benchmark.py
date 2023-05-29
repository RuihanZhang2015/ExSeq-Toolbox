import os
import pickle
import plotly.subplots as sp
import plotly.graph_objs as go
from exm.utils import retrieve_all_puncta


def puncta_analysis(args, fov, improved=False):

    if improved:
        with open(args.work_path + '/fov{}/improved_puncta_results.pickle'.format(fov), 'rb') as f:
            consolidated_puncta = pickle.load(f)
    else:
        consolidated_puncta = retrieve_all_puncta(args, fov)

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

    with open(args.work_path + file_name, 'wb') as f:
        pickle.dump(result, f)


def plot_benchmark_data(args, fov):

    with open(args.work_path + 'fov{}/original_puncta_analysis.pickle'.format(fov), 'rb') as f:
        puncta_original = pickle.load(f)

    with open(args.work_path + 'fov{}/improved_puncta_analysis.pickle'.format(fov), 'rb') as f:
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
                        'Distribution of Missing Codes',
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

    fig.update_layout(height=1500, width=1300, barmode='group',
                      title_text="Puncta Extraction Summary (FOV {})".format(fov))
    
    fig.write_html(os.path.join(
        args.work_path, 'fov{}/puncta_comparison_dashboard_fov_{}.html'.format(fov)))
    fig.show()
