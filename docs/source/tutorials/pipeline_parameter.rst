1- Pipeline Parameter Configuration
====================================

This step focuses on configuring the pipeline parameters for the ExSeq-Toolbox. The process involves setting up the logger, initializing configuration objects, and defining mandatory and optional configuration parameters.

Step 1: Import Modules and Configure Logger
-------------------------------------------

First, import the necessary modules and configure the logger:

.. code-block:: python

    from exm.utils.log import configure_logger
    from exm.args.args import Args

    # Configure logger for ExSeq-Toolbox
    logger = configure_logger('ExSeq-Toolbox')

Step 2: Initialize Configuration Object
---------------------------------------

Initialize the configuration object to hold all pipeline parameters:

.. code-block:: python

    # Initialize the configuration object.
    args = Args()

Step 3: Mandatory Configuration
-------------------------------

Set the absolute path to the raw data directory. This is a required setting:

.. code-block:: python

    raw_data_directory = '/path/to/your/raw_data_directory/'

Step 4: Raw Data Directory Structure
------------------------------------

The ExSeq-Toolbox assumes a specific directory structure for raw data:

.. code-block:: text

    # Ensure that your raw data adheres to this directory structure before running the package.
    # raw_data_directory/
        # ├── code0/
        # │   ├── Channel405 SD_Seq0004.nd2
        # │   ├── Channel488 SD_Seq0003.nd2
        # │   ├── Channel561 SD_Seq0002.nd2
        # │   ├── Channel594 SD_Seq0001.nd2
        # │   ├── Channel604 SD_Seq0000.nd2
        # ├── code1/
        # │   ├── Channel405 SD_Seq0004.nd2
        # │   ├── ...
        # ├── ...

Step 5: Optional Configuration
------------------------------

Define additional optional configuration parameters like `codes_list`, `fov_list`, and others:

.. code-block:: python

    codes_list = list(range(7))
    fov_list = list(range(12))  # Example values
    ...

Step 6: Set Parameters
----------------------

Finally, set the parameters using the `set_params` method of the `args` object:

.. code-block:: python

    args.set_params(
        raw_data_path=raw_data_directory,
        ...
        args_file_name=args_file
    )

    # Note: Always ensure that the paths and other configuration parameters are correct before running the script.

Next Steps
----------

After configuring the pipeline parameters, the next step is to proceed to *Volume Alignment*. For details on how to perform volume alignment, refer to the `Volume Alignment <volume_alignment.html>`_ section of this guide.
