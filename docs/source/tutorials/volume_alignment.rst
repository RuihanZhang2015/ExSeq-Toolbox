.. _volume-alignment-section:

2- Volume Alignment
====================

This section covers the process of executing volumetric alignment in the ExSeq-Toolbox. This step is crucial for ensuring accurate alignment of data across different imaging rounds and fields of view (FOVs).

Load the Configuration
----------------------

Start by loading the configuration settings for the toolbox.

.. code-block:: python

    from exm.args.args import Args

    # Initialize the configuration object.
    args = Args()

    # Provide the path to the configuration file.
    args_file_path = '/path/to/your/parameters/file.json'

    # Load the configuration settings from the specified file.
    args.load_params(args_file_path)

    # Note: Ensure the provided path points to the correct configuration file.

Execute Volumetric Alignment
----------------------------

Execute the volumetric alignment using the specified parameters.

.. code-block:: python

    from exm.align.align import volumetric_alignment

    # Specify additional parameters for alignment
    parallelization = 4  # Number of parallel processes
    alignment_method = 'bigstream'  # or None for SimpleITK
    background_subtraction = ''  # 'top_hat' or 'rolling_ball'

    # Specific round and ROI pairs for alignment
    specific_code_fov_pairs = [(code_val, fov_val) for code_val in args.codes for fov_val in args.fovs]

    volumetric_alignment(
        args=args,
        code_fov_pairs=specific_code_fov_pairs,
        parallel_processes=parallelization,
        method=alignment_method,
        bg_sub=background_subtraction
    )

    # Note: Monitor the logs or console output for the alignment process.

Next Steps
----------

After completing the volumetric alignment, the next step in the pipeline is *Alignment Evaluation*. This step involves assessing the quality and accuracy of the alignment to ensure that the data is correctly aligned across different imaging rounds and fields of view. For detailed instructions on how to perform alignment evaluation, see the `Alignment Evaluation <alignment_evaluation.html>`_ section of this guide.
