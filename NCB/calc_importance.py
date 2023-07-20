import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2 as svox2split
import json
import os
import numpy as np
import argparse
from util.dataset import datasets
from util import config_util
from warnings import warn
from datetime import datetime

from tqdm import tqdm

def input_mapping(x, B): 
    if B is None:
        return x
    else:
        x_proj = (2.*np.pi*x) @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)

def get_coord_table(coord_shape, device='cuda', dtype=torch.float32):
    assert len(coord_shape) in [1, 2, 3]
    if len(coord_shape) == 1:
        W = coord_shape
        xx = torch.meshgrid(
            torch.arange(W, dtype=dtype, device=device) + 0.5,
        )
        xx = (xx-(W//2)) / (W//2)
        coord_table = torch.stack([xx], -1)
    elif len(coord_shape) == 2:
        H, W = coord_shape
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=dtype, device=device) + 0.5,
            torch.arange(W, dtype=dtype, device=device) + 0.5,
        )
        xx = (xx-(W//2)) / (W//2)
        yy = (yy-(H//2)) / (H//2) 
        coord_table = torch.stack([xx, yy], -1)
        
    elif len(coord_shape) == 3:
        D, H, W = coord_shape
        zz, yy, xx = torch.meshgrid(
            torch.arange(D, dtype=dtype, device=device) + 0.5,
            torch.arange(H, dtype=dtype, device=device) + 0.5,
            torch.arange(W, dtype=dtype, device=device) + 0.5,
        )
        xx = (xx-(W//2)) / (W//2)
        yy = (yy-(H//2)) / (H//2) 
        zz = (zz-(D//2)) / (D//2) 
        coord_table = torch.stack([xx, yy, zz], -1)
    else:
        raise NotImplementedError
        
    return coord_table.to(device).type(dtype).unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
config_util.define_common_args(parser)

group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')

group.add_argument('--reso',
                        type=str,
                        default=
                        "[[256, 256, 256], [512, 512, 512]]",
                       help='List of grid resolution (will be evaled as json);'
                            'resamples to the next one every upsamp_every iters, then ' +
                            'stays at the last one; ' +
                            'should be a list where each item is a list of 3 ints or an int')
group.add_argument('--upsamp_every', type=int, default=
                     3 * 12800,
                    help='upsample the grid every x iters')
group.add_argument('--init_iters', type=int, default=
                     0,
                    help='do not upsample for first x iters')
group.add_argument('--upsample_density_add', type=float, default=
                    0.0,
                    help='add the remaining density by this amount when upsampling')

group.add_argument('--basis_type',
                    choices=['sh', '3d_texture', 'mlp'],
                    default='sh',
                    help='Basis function type')

group.add_argument('--basis_reso', type=int, default=32,
                   help='basis grid resolution (only for learned texture)')
group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

group.add_argument('--mlp_posenc_size', type=int, default=4, help='Positional encoding size if using MLP basis; 0 to disable')
group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

group.add_argument('--background_nlayers', type=int, default=0,#32,
                   help='Number of background layers (0=disable BG model)')
group.add_argument('--background_reso', type=int, default=512, help='Background resolution')



group = parser.add_argument_group("optimization")
group.add_argument('--n_iters', type=int, default=10 * 12800, help='total number of iters to optimize for')
group.add_argument('--batch_size', type=int, default=
                     5000,
                     #100000,
                     #  2000,
                   help='batch size')


# TODO: make the lr higher near the end
group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)


group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
group.add_argument('--lr_sh', type=float, default=
                    1e-2,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float,
                      default=
                    5e-6
                    )
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

# BG LRs
group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_color_bg', type=float, default=1e-1,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
# END BG LRs

group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
group.add_argument('--lr_basis', type=float, default=#2e6,
                      1e-6,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_basis_final', type=float,
                      default=
                      1e-6
                    )
group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
group.add_argument('--lr_basis_delay_steps', type=int, default=0,#15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_basis_begin_step', type=int, default=0)#4 * 12800)
group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--save_every', type=int, default=5,
                   help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')
group.add_argument('--init_sigma_bg', type=float,
                   default=0.1,
                   help='initialization sigma (for BG)')

# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=False)
group.add_argument('--log_depth_map', action='store_true', default=False)
group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
        help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


group = parser.add_argument_group("misc experiments")
group.add_argument('--thresh_type',
                    choices=["weight", "sigma"],
                    default="weight",
                   help='Upsample threshold type')
group.add_argument('--weight_thresh', type=float,
                    default=0.0005 * 512,
                    #  default=0.025 * 512,
                   help='Upsample weight threshold; will be divided by resulting z-resolution')
group.add_argument('--density_thresh', type=float,
                    default=5.0,
                   help='Upsample sigma threshold')
group.add_argument('--background_density_thresh', type=float,
                    default=1.0+1e-9,
                   help='Background sigma threshold for sparsification')
group.add_argument('--max_grid_elements', type=int,
                    default=44_000_000,
                   help='Max items to store after upsampling '
                        '(the number here is given for 22GB memory)')

group.add_argument('--tune_mode', action='store_true', default=False,
                   help='hypertuning mode (do not save, for speed)')
group.add_argument('--tune_nosave', action='store_true', default=False,
                   help='do not save any checkpoint even at the end')


group = parser.add_argument_group("losses")
# Foreground TV
group.add_argument('--lambda_tv', type=float, default=1e-5)
group.add_argument('--tv_sparsity', type=float, default=0.01)
group.add_argument('--tv_logalpha', action='store_true', default=False,
                   help='Use log(1-exp(-delta * sigma)) as in neural volumes')

group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)#1e-2)#1e-3)
group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

group.add_argument('--tv_decay', type=float, default=1.0)

group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")

group.add_argument('--tv_contiguous', type=int, default=1,
                        help="Apply TV only on contiguous link chunks, which is faster")
# End Foreground TV

group.add_argument('--lambda_sparsity', type=float, default=
                    0.0,
                    help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                         "(but applied on the ray)")
group.add_argument('--lambda_beta', type=float, default=
                    0.0,
                    help="Weight for beta distribution sparsity loss as in neural volumes")


# Background TV
group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-2)
group.add_argument('--lambda_tv_background_color', type=float, default=1e-2)

group.add_argument('--tv_background_sparsity', type=float, default=0.01)
# End Background TV

# Basis TV
group.add_argument('--lambda_tv_basis', type=float, default=0.0,
                   help='Learned basis total variation loss')
# End Basis TV

group.add_argument('--weight_decay_sigma', type=float, default=1.0)
group.add_argument('--weight_decay_sh', type=float, default=1.0)

group.add_argument('--lr_decay', action='store_true', default=True)

group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

group.add_argument('--nosphereinit', action='store_true', default=False,
                     help='do not start with sphere bounds (please do not use for 360)')

group.add_argument('--pretrained', type=str, default=None,
                    help='pretrained model')

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

os.makedirs(args.train_dir, exist_ok=True)

torch.manual_seed(20200823)
np.random.seed(20200823)

factor = 1
dset = datasets[args.dataset_type](
               args.data_dir,
               split="train",
               device=device,
               factor=factor,
               n_images=args.n_train,
               **config_util.build_data_options(args))

if args.background_nlayers > 0 and not dset.should_use_background:
    warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

global_start_time = datetime.now()
print('============================', args.pretrained)

assert os.path.exists(args.pretrained)

print("Loaded pretrained model:", args.pretrained)
grid = svox2split.SparseGrid.load(args.pretrained, device=device)
train_from_scratch = False

grid.requires_grad_(True)
config_util.setup_render_opts(grid.opt, args)
print('Render options', grid.opt)

gstep_id_base = 0

grid.d_mean = grid.density_data.mean(0)
grid.d_std = grid.density_data.std(0)
grid.sh_mean = grid.sh_data.mean(0)
grid.sh_std = grid.sh_data.std(0)

print('initialize the loss weight by caculating mse of each ray from pretrained model')
batchsize = 800 * 40

pbar = tqdm(enumerate(range(0, len(dset.rays_init), batchsize)), total=len(dset.rays_init)//batchsize)
pbar.set_description("init_mse_weight")
flag = 0
with Timing("all"):
    for _, batch_begin in pbar:
        batch_end = batch_begin + batchsize
        batch_origins = dset.rays_init.origins[batch_begin: batch_end].to(device)
        batch_dirs = dset.rays_init.dirs[batch_begin: batch_end].to(device)
        rgb_gt = dset.rays_init.gt[batch_begin: batch_end].to(device)
        batch_views = torch.randint_like(batch_dirs, high=18,low=0)[:,0].contiguous().int()
        rays = svox2split.Rays(batch_origins, batch_dirs, batch_views)

        rgb_pred,importance,sigma, fixed_part = grid._volume_render_importance(rays)
        
        (importance * sigma).sum().backward()
        
        flag += 1


grad_sigma,_,_,_ = grid._get_data_grads()
grid.density_data.grad = None
grid.sh_data.grad = None

save_dir = os.path.dirname(args.pretrained)
torch.save(grad_sigma, os.path.join(save_dir, 'importance_map.pth'))
print('save to ', os.path.join(save_dir, 'importance_map.pth'))