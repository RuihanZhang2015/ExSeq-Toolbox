import h5py
import numpy as np
from exm.segmentation.segmentation import segment_3d
from cellpose import models
import os
import tifffile
from exm.args.args import Args
import torch
print("Using GPU:", torch.cuda.is_available())

# Step 1: Load Configuration Settings
# ====================================

# Initialize the Args object for configuration.
args = Args()

# Provide the path to the configuration file.
args_file_path = '/orcd/data/edboyden/002/davy/ExSeq-Toolbox/ExSeq_toolbox_args.json'
# Load the configuration settings from the file.
args.load_params(args_file_path)

# Loop through each FOV
model = models.CellposeModel(gpu=True)

for fov in range(len(args.fovs)):
    # Construct input file path
    h5_path = f'/orcd/data/edboyden/002/davy/ExSeq-Toolbox/processed/code0/{fov}.h5'
    
    # Load volume from HDF5
    with h5py.File(h5_path, 'r') as f:
        volume = np.array(f['405'])  # adjust key as needed
    # Run segmentation
    print("start segmentation")
    masks = segment_3d(volume, model,downsample=True)

    # Construct output path
    save_path = os.path.join(args.puncta_path, f"nuclei_segmentation/fov{fov}_mask.tif")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save mask
    tifffile.imwrite(save_path, masks.astype(np.uint16))