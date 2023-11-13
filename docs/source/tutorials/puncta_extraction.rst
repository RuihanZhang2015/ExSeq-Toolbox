.. _puncta-extraction-section:

4- Puncta Extraction
=====================

This section describes the process of puncta extraction in the ExSeq-Toolbox. It involves extracting puncta from images, consolidating channel information, and preparing data for subsequent analysis steps.

Step 1: Load Configuration Settings
------------------------------------

Begin by initializing and loading the configuration settings.

.. code-block:: python

    from exm.args.args import Args

    # Initialize the Args object for configuration.
    args = Args()

    # Provide the path to the configuration file.
    args_file_path = '/path/to/your/parameter/file.json'

    # Load the configuration settings from the file.
    args.load_params(args_file_path)

    # Ensure the correct configuration file path.

Step 2: Puncta Extraction
-------------------------

Extract puncta using specified parameters and configuration settings.

.. code-block:: python

    from exm.puncta.extract import extract

    # Define the list of Code, FOV pairs for extraction.
    code_fov_pairs_to_extract = [(code_val, fov_val) for code_val in args.codes for fov_val in args.fovs]

    # Parameters for extraction.
    use_gpu_setting = True
    num_gpus = 3
    num_cpus = 3

    extract(args=args,
            code_fov_pairs=code_fov_pairs_to_extract,
            use_gpu=use_gpu_setting,
            num_gpu=num_gpus,
            num_cpu=num_cpus)

Step 3: Channel Consolidation
-----------------------------

Consolidate extracted channel information.

.. code-block:: python

    from exm.puncta.consolidate import consolidate_channels

    # Parameters for channel consolidation.
    num_cpus_for_channel = 4

    consolidate_channels(args=args,
                         code_fov_pairs=code_fov_pairs_to_extract,
                         num_cpu=num_cpus_for_channel)

Step 4: Code Consolidation
--------------------------

Consolidate codes for the analysis.

.. code-block:: python

    from exm.puncta.consolidate import consolidate_codes

    # Parameters for code consolidation.
    fov_list_to_consolidate = list(args.fovs)
    num_cpus_for_code = 4

    consolidate_codes(args=args, fov_list=fov_list_to_consolidate, num_cpu=num_cpus_for_code)

    # Monitor logs or console output for processing errors.

Next Steps
----------

Following the completion of puncta extraction and consolidation, the next phase in the ExSeq-Toolbox workflow is *Puncta Basecalling*. This step is vital for interpreting the extracted data in biological context. For detailed instructions on Puncta Basecalling, refer to the `Puncta Basecalling <puncta_basecalling.html>`_ section of this guide.
