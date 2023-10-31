import os
import h5py
import shutil
import numpy as np

def setup_dataset():
    # Define the directories
    base_dir = os.path.join(os.environ.get("BASE_DIR"),'dataset')

    dir_names = ["code0", "code1", "code2"]

    # Create directories
    for dir_name in dir_names:
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)


    # Function to create the volume with a white center and optional shift
    def create_volume(shift_x=0, shift_y=0, shift_z=0):
        volume = np.zeros((10, 10, 10), dtype=np.uint16)
        volume[4+shift_x:7+shift_x, 4+shift_y:7+shift_y, 4+shift_z:7+shift_z] = 65535
        return volume

    # Create .h5 files with the modified properties
    shifts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]  # No shift, shift in x, shift in y
    channels = ['640','594','561','488','405']
    for dir_index, dir_name in enumerate(dir_names):
        for i,channel in zip(list(range(5)),channels):
            file_path = os.path.join(base_dir, f"code{dir_index}/Channel{channel} SD_Seq000{i}.h5")
            
            # Create the volume with the appropriate shift
            volume = create_volume(*shifts[dir_index])
            
            # Save the volume in the .h5 file
            with h5py.File(file_path, 'w') as f:
                dataset_name = f"{channel}"
                f.create_dataset(dataset_name, data=volume)

def cleanup_dataset():
    base_dir = os.path.join(os.environ.get("BASE_DIR"),'dataset')
    shutil.rmtree(base_dir, ignore_errors=True)

def pytest_sessionstart(session):
    setup_dataset()

def pytest_sessionfinish(session, exitstatus):
    cleanup_dataset()