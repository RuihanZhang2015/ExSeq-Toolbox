.. _puncta-basecalling-section:

5- Puncta Basecalling
======================

"Puncta Basecalling" is the final step in the ExSeq-Toolbox data processing pipeline. This step involves assigning genes to detected puncta and linking puncta to nuclei within the fields of view (FOVs).

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

Step 2: Assign Genes to Detected Puncta for All FOVs
----------------------------------------------------

Assign genes to detected puncta across all specified FOVs.

.. code-block:: python

    from exm.puncta.basecalling import puncta_assign_gene

    operation_mode = 'original'  # Can be 'original' or 'improved'.

    for fov_for_gene_assignment in args.fovs:
        puncta_assign_gene(args=args, 
                           fov=fov_for_gene_assignment, 
                           option=operation_mode)

Step 3: Assign Puncta to Nuclei for All FOVs
--------------------------------------------

Assign puncta to nearest nuclei based on specified parameters.

.. code-block:: python

    from exm.puncta.basecalling import puncta_assign_nuclei

    # Parameters for assigning puncta to nuclei.
    distance_thresh = 100
    compare_to_surface = True
    nearest_nuclei_count = 3

    for fov_for_nuclei_assignment in args.fovs:
        puncta_assign_nuclei(args=args, 
                             fov=fov_for_nuclei_assignment, 
                             distance_threshold=distance_thresh, 
                             compare_to_nuclei_surface=compare_to_surface, 
                             num_nearest_nuclei=nearest_nuclei_count, 
                             option=operation_mode)

    # Monitor logs or console output for processing errors.

Conclusion
----------

With the completion of the Puncta Basecalling step, the data processing pipeline of the ExSeq-Toolbox is concluded. You should now have a fully processed dataset ready for further analysis or interpretation.
