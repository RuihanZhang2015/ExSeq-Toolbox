from exm.args import Args
project_path = '/mp/nas3/ruihan/20230308_celegans/'

# init args 
args = Args()
# set args 
args.set_params(project_path, pickle_file = 'args_ruihan.pkl')

# load args 
# args.load_params('/mp/nas3/ruihan/20221218_zebrafish/args.pkl')

# args.print()
# args.tree()

from exm.align import correlation_lags
# code_fov_pairs = [[2,fov] for fov in args.fovs]
# lag_dict = correlation_lags(args,code_fov_pairs)
# code_fov_pairs = [[3,fov] for fov in args.fovs]
# lag_dict = correlation_lags(args,code_fov_pairs)

from exm.align import inspect_align_truncated
code_fov_pairs = [[1,fov] for fov in args.fovs]
inspect_align_truncated(args, code_fov_pairs)