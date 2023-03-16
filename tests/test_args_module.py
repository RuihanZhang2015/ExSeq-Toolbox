import os
from exm.args import Args
from exm.align import transform_ref_code,transform_other_code
from exm.puncta.extract import extract
from exm.puncta.consolidate import consolidate_channels,consolidate_codes

def test_args_module():
    project_path = '/mp/nas3/ruihan/20221218_zebrafish/'
    align_z_init = '/home/mansour/ExSeq-Toolbox/examples/align_z_init.pkl'
    args = Args() 
    args.set_params(project_path,align_z_init=align_z_init)
    args_path = os.path.join(project_path,'args.pkl')

    assert os.path.exists(args_path) == True


def test_align_module_transform_ref_code():
    args = Args()
    args.load_params('/mp/nas3/ruihan/20221218_zebrafish/args.pkl')
    code_fov_pairs = [[args.ref_code,0]]
    transform_ref_code(args,code_fov_pairs,mode = 'all')
    output_path = args.h5_path.format(args.ref_code,0)
    assert os.path.exists(output_path) == True


def test_align_module_transform_other_function():
    args = Args()
    args.load_params('/mp/nas3/ruihan/20221218_zebrafish/args.pkl')
    code_fov_pairs = [[1,0]]
    transform_other_code(args,code_fov_pairs,mode = 'all')
    output_path = args.h5_path.format(1,0)
    assert os.path.exists(output_path) == True

def test_puncta_module_extract():
    args = Args()
    args.load_params('/mp/nas3/ruihan/20221218_zebrafish/args.pkl')
    code_fov_pairs = [[0,0],[1,0]]
    extract(args, code_fov_pairs, use_gpu = True,num_cpu=8)

    assert os.path.exists(os.path.join(args.work_path,'fov{}/coords_total_code{}.pkl'.format(0,0))) == True
    assert os.path.exists(os.path.join(args.work_path,'fov{}/coords_total_code{}.pkl'.format(0,1))) == True

def test_puncta_module_consolidate_channels():
    args = Args()
    args.load_params('/mp/nas3/ruihan/20221218_zebrafish/args.pkl')
    code_fov_pairs = [[0,0],[1,0]]
    consolidate_channels(args,code_fov_pairs)

    assert os.path.exists(os.path.join(args.work_path +'/fov{}/result_code{}.pkl'.format(0,0))) == True
    assert os.path.exists(os.path.join(args.work_path +'/fov{}/result_code{}.pkl'.format(0,1))) == True


def test_puncta_module_consolidate_codes():
    args = Args()
    args.load_params('/mp/nas3/ruihan/20221218_zebrafish/args.pkl')
    fov_list = [0,1]
    consolidate_codes(args,fov_list)

    assert os.path.exists(os.path.join(args.work_path,'fov{}/result.pkl'.format(0))) == True
    assert os.path.exists(os.path.join(args.work_path,'fov{}/result.pkl'.format(1))) == True





