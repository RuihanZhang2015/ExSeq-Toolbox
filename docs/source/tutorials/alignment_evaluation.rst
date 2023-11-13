.. _alignment-evaluation-section:

3- Alignment Evaluation
========================

This section guides you through the process of evaluating the alignment in the ExSeq-Toolbox. It involves measuring alignment accuracy and calculating confidence intervals to ensure reliable data processing.

Step 1: Load Configuration Settings
------------------------------------

Begin by loading the configuration settings for the toolbox.

.. code-block:: python

    from exm.args.args import Args

    # Create a new Config object instance.
    args = Args()

    # Provide the path to the configuration file.
    args_file_path = '/path/to/your/parameter/file.json'

    # Load the configuration settings from the specified file.
    args.load_config(args_file_path)

Step 2: Additional Configuration for Alignment Evaluation
----------------------------------------------------------

Configure additional parameters specific to alignment evaluation.

.. code-block:: python

    # Set various parameters for alignment evaluation
    args.nonzero_thresh = .2 * 2048 * 2048 * 80
    args.N = 1000
    args.subvol_dim = 100
    args.xystep = 0.1625/40  # check value
    args.zstep = 0.4/40  # check value
    args.pct_thresh = 99

Step 3: Alignment Measurement
-----------------------------

Measure the alignment for specified codes and FOVs.

.. code-block:: python

    from exm.align.align_eval import measure_round_alignment_NCC

    # Define the list of Codes and Fovs for alignment evaluation
    codes_to_analyze = args.codes
    fovs_to_analyze = args.fovs  # e.g., [1, 3].

    # Extract the coordinates and measure alignment
    for fov in fovs_to_analyze:
        for code in codes_to_analyze:
            measure_round_alignment_NCC(args=args, code=code, fov=fov)

Step 4: Alignment Evaluation and Confidence Interval Calculation
----------------------------------------------------------------

Evaluate alignment accuracy and calculate confidence intervals.

.. code-block:: python

    from exm.align.align_eval import plot_alignment_evaluation, calculate_alignment_evaluation_ci

    # Define CI and percentile parameters
    ci_percentage = 95
    percentile_filter_value = 95

    for fov in fovs_to_analyze:
        # Plot and calculate CI for alignment evaluation
        plot_alignment_evaluation(args, fov, percentile=percentile_filter_value, save_fig=True)
        calculate_alignment_evaluation_ci(args, fov, ci=ci_percentage, percentile_filter=percentile_filter_value)

Next Steps
----------

After assessing the alignment, the subsequent step in the ExSeq-Toolbox pipeline is *Puncta Extraction*. This step is crucial for identifying and analyzing specific biological structures in the data. For detailed instructions on Puncta Extraction, refer to the `Puncta Extraction <puncta_extraction.html>`_ section of this guide.
