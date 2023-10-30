import os
import pytest
import numpy as np
from exm.args.args import Args
from exm.align.align import volumetric_alignment
from exm.puncta.extract import extract
from exm.puncta.consolidate import consolidate_channels, consolidate_codes


def test_args_module():

    args = Args()
    raw_data_directory = os.path.join(os.environ.get("BASE_DIR"),'dataset')
    codes_list = list(range(3))
    fov_list = list(range(1))  
    processed_data_directory = os.path.join(raw_data_directory,'processed_data')
    pixel_spacing = [0.4, 1.625, 1.625]
    channel_names_list = ['640', '594', '561', '488', '405']
    reference_code = 0
    reference_channel = '405'
    gene_digit_file = './gene_list.csv'
    permissions_flag = False
    create_directory_structure_flag = True
    args_file = "ExSeq_toolbox_args"

    args.set_params(
        raw_data_path=raw_data_directory,
        processed_data_path=processed_data_directory,
        codes=codes_list,
        fovs=fov_list,
        spacing=pixel_spacing,
        channel_names=channel_names_list,
        ref_code=reference_code,
        ref_channel=reference_channel,
        gene_digit_csv=gene_digit_file,
        permission=permissions_flag,
        create_directroy_structure=create_directory_structure_flag,
        args_file_name=args_file
    )
    
    args_path = os.path.join(processed_data_directory, "ExSeq_toolbox_args.json")
    assert os.path.exists(args_path), f"Expected {args_path} to exist, but it doesn't."


def test_align_module_bigstream():
    args = Args()
    raw_data_directory = os.path.join(os.environ.get("BASE_DIR"),'dataset')
    args_file_path = os.path.join(raw_data_directory,'processed_data','ExSeq_toolbox_args.json')
    args.load_params(args_file_path)
    code_fov_pairs = [[0,0],[1,0],[2,0]]
    parallelization = 1
    alignment_method = 'bigstream'  
    background_subtraction = ''  
    volumetric_alignment(
        args=args,
        code_fov_pairs=code_fov_pairs,
        parallel_processes=parallelization,
        method=alignment_method,
        bg_sub=background_subtraction,
        dataset_type='.h5'
    )
    
    assert os.path.exists(args.h5_path.format(0, 0)), f"Expected {args.h5_path.format(0, 0)} to exist, but it doesn't."
    assert os.path.exists(args.h5_path.format(1, 0)), f"Expected {args.h5_path.format(1, 0)} to exist, but it doesn't."
    assert os.path.exists(args.h5_path.format(2, 0)), f"Expected {args.h5_path.format(2, 0)} to exist, but it doesn't."


def test_puncta_module_extract():
    args = Args()
    raw_data_directory = os.path.join(os.environ.get("BASE_DIR"),'dataset')
    args_file_path = os.path.join(raw_data_directory,'processed_data','ExSeq_toolbox_args.json')
    args.load_params(args_file_path)
    code_fov_pairs = [[0,0],[1,0],[2,0]]
    extract(args, code_fov_pairs, use_gpu=False, num_cpu=1)

    assert os.path.exists(os.path.join(args.puncta_path, "fov{}/coords_total_code{}.pkl".format(0, 0))), f"Expected {os.path.join(args.puncta_path, 'fov{}/coords_total_code{}.pkl'.format(0, 0))} to exist, but it doesn't."
    assert os.path.exists(os.path.join(args.puncta_path, "fov{}/coords_total_code{}.pkl".format(0, 1))), f"Expected {os.path.join(args.puncta_path, 'fov{}/coords_total_code{}.pkl'.format(0, 1))} to exist, but it doesn't."


def test_puncta_module_consolidate_channels():
    args = Args()
    raw_data_directory = os.path.join(os.environ.get("BASE_DIR"),'dataset')
    args_file_path = os.path.join(raw_data_directory,'processed_data','ExSeq_toolbox_args.json')
    args.load_params(args_file_path)
    code_fov_pairs = [[0,0],[1,0],[2,0]]
    consolidate_channels(args, code_fov_pairs,num_cpu=1)

    assert os.path.exists(os.path.join(args.puncta_path + "fov{}/result_code{}.pkl".format(0, 0))), f"Expected {os.path.join(args.puncta_path + 'fov{}/result_code{}.pkl'.format(0, 0))} to exist, but it doesn't."
    assert os.path.exists(os.path.join(args.puncta_path + "fov{}/result_code{}.pkl".format(0, 1))), f"Expected {os.path.join(args.puncta_path + 'fov{}/result_code{}.pkl'.format(0, 1))} to exist, but it doesn't." 


def test_puncta_module_consolidate_codes():
    args = Args()
    raw_data_directory = os.path.join(os.environ.get("BASE_DIR"),'dataset')
    args_file_path = os.path.join(raw_data_directory,'processed_data','ExSeq_toolbox_args.json')
    args.load_params(args_file_path)
    fov_list = [0]
    consolidate_codes(args, fov_list,num_cpu=1)

    assert os.path.exists(os.path.join(args.puncta_path, "fov{}/result.pkl".format(0))) ,f"Expected {os.path.join(args.puncta_path, 'fov{}/result.pkl'.format(0))} to exist, but it doesn't."

