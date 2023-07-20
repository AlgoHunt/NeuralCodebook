import argparse
import svox2
import torch
import os
import numpy as np
from torch import nn
from tqdm import tqdm

def extract_sh_and_density(grid):
    mask = grid.links.view(-1) >= 0
    full_density = torch.zeros(mask.shape[0], 1).float().to(device)
    full_density[mask] = grid.density_data
    full_density = full_density.view(*grid.links.shape, 1)

    full_sh = torch.zeros(mask.shape[0], 27).float().to(device)
    full_sh[mask] = grid.sh_data
    full_sh = full_sh.view(*grid.links.shape, 27)
    return full_sh, full_density

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

def make_network(num_layers, num_channels, input_channels=None, output_channels=3):
    if input_channels is None:
        input_channels = num_channels
    layers = []
    layers.append(nn.Linear(input_channels,num_channels))
    layers.append(nn.LeakyReLU(0.2))
    for i in range(num_layers-2):
        layers.append(nn.Linear(num_channels,num_channels))
        layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.Linear(num_channels,output_channels))
    return nn.Sequential(*layers)

def input_mapping(x, B): 
    if B is None:
        return x
    else:
        x_proj = (2.*np.pi*x) @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)


def define_common_args():
    parser = argparse.ArgumentParser("Neural Code Book training")
    train = parser.add_argument_group("train")
    train.add_argument("sr_ckpt", type=str, help="the input super-resolution ckpt file")
    train.add_argument("hr_ckpt", type=str, help="the input high-resolution ckpt file")
    train.add_argument("data_dir", type=str, help="dataset directory")
    train.add_argument("--lr_ckpt", type=str, help="the input low-resolution ckpt file")
    train.add_argument("--save_dir","-t", type=str, default="ncb", help="checkpoint and logging directory")

    train.add_argument("--optim", type=str, default='adam', help='optimizer config')
    train.add_argument("--n_iters", type=int, default=20000)
    train.add_argument("--batch_size", type=int, default=100000, help='batch size')
    train.add_argument("--lr_density", type=float, default=5e-3, help="lr for density code book")
    train.add_argument("--lr_sh", type=float, default=5e-3, help="lr for sh code book")

    train.add_argument("--weight_decay_density", type=float, default=1e-5, help="weight decay for density code book")
    train.add_argument("--weight_decay_sh", type=float, default=1e-5, help="weight decay for sh code book")
    train.add_argument("--gpu", "-g" ,type=int, default=0, help="gpu num")
    train.add_argument("--use_lowres", action="store_true", help="load lr model as input too")

    train.add_argument("--importance_map_path", action='store_true', help="load lr model as input too")
    train.add_argument("-c", "--config", type=str, default='configs/llff.json', help="load lr model as input too")
    train.add_argument("--use_importance_threshold", action='store_true', help="load lr model as input too")
    train.add_argument("--threshold_percent", type=float,default=0.05, help="load lr model as input too")

    group_mlp = parser.add_argument_group("mlp")
    group_mlp.add_argument("--layer_density", type=int, default=7, help="density mlp layers num")
    group_mlp.add_argument("--channel_density", type=int, default=256, help="density mlp channel num")
    group_mlp.add_argument("--layer_sh", type=int, default=8, help="sh mlp layers num")
    group_mlp.add_argument("--channel_sh", type=int, default=512, help="sh mlp channel num")
    group_mlp.add_argument("--pos_emb", type=int, default=128, help="pos embedding length")
    
    return parser.parse_args()

args = define_common_args()

save_dir = args.save_dir
save_file = os.path.join(save_dir,'tune.npz')
# if os.path.isfile(save_file):
#     print("SKIP training!")
#     if not os.path.isfile(os.path.join(save_dir, 'result.txt')):
#         print("Run eval!")
#         os.system(f"python eval.py  {save_file}  {args.data_dir} -c {args.config}")
#     else:
#         print('Skip evaling!')
#     exit()
device = f"cuda:{args.gpu}"
grid_input = svox2.SparseGrid.load(args.sr_ckpt, device=device)
grid_target = svox2.SparseGrid.load(args.hr_ckpt,device=device)

mask_input = grid_input.links >= 0
mask_target = grid_target.links >= 0
assert mask_input.sum() == mask_target.sum(), f'{mask_input.sum()} vs {mask_target.sum()}, sr`s mask should equal to target`s mask'

reso = grid_input.links.shape

os.makedirs(save_dir, exist_ok=True)

if args.importance_map_path:
    importance_map_path = os.path.join(os.path.dirname(args.hr_ckpt), 'importance_map.pth')
    print("loading importance map from ", importance_map_path)
    importance_map = torch.load(importance_map_path).to(device)
    fac = importance_map.sum()/importance_map.numel()
    importance_map = importance_map/fac


with torch.no_grad():
    all_input = extract_sh_and_density(grid_input)
    all_target = extract_sh_and_density(grid_target)

    all_input = torch.cat(all_input, -1).reshape(-1,28)
    all_target = torch.cat(all_target, -1).reshape(-1,28)
    mask = (grid_input.links>=0).reshape(-1)
    all_input = all_input[mask,:]
    all_target = all_target[mask,:]
    
    sh_input, d_input = all_input[:,:27],all_input[:,27:]
    sh_target, d_target = all_target[:,:27],all_target[:,27:]
    
    d_mean = d_input.mean(0)
    d_std = d_input.std(0)
    sh_mean = sh_input.mean(0)
    sh_std = sh_input.std(0)
    
    d_target = (d_target - d_mean)/d_std
    sh_target = (sh_target - sh_mean)/sh_std
    
    d_input = (d_input - d_mean)/d_std
    sh_input = (sh_input - sh_mean)/sh_std


    loss_sh = .5 * torch.mean((sh_input - sh_target) ** 2)
    loss_density = .5 * torch.mean((d_target - d_input) ** 2)
    print(loss_sh, loss_density)
    print(d_mean, d_std, sh_mean, sh_std)

    del all_input
    del grid_target
    del all_target

if args.use_lowres:
    with torch.no_grad():
        grid_lowres = svox2.SparseGrid.load(args.lr_ckpt, device=device)
        all_lowres = extract_sh_and_density(grid_lowres)
        all_lowres = torch.cat(all_lowres, -1).reshape(-1,28)
    
        mask = (grid_lowres.links>=0).reshape(-1)
        assert mask.sum() == mask_input.sum(), "lowres`s mask should equal to input`s mask"
        

        all_lowres = all_lowres[mask,:]
        
        sh_lowres, d_lowres = all_lowres[:,:27],all_lowres[:,27:]
        

        d_lowres = (d_lowres - d_mean)/d_std
        sh_lowres = (sh_lowres - sh_mean)/sh_std

        del grid_lowres
        del all_lowres

print("Pass 1/2 density codebook training: ")
# ---- density codebook training
pos_emb = args.pos_emb                  # 128
B_density = torch.normal(mean=0, std=1, size=(pos_emb, 3)).to(device) * 10
xyz_table = get_coord_table(coord_shape=reso,device=device).reshape(-1,3)
xyz_table = xyz_table[mask,:]
input_channels = pos_emb*2 + 27 + 1

    
CodebookNetDensity = make_network(
    args.layer_density, 
    args.channel_density, 
    input_channels=input_channels ,
    output_channels=1).to(device)



# ---- sh codebook training
print("Pass 2/2 sh codebook training: ")
pos_emb = args.pos_emb  
B_sh = torch.normal(mean=0, std=1, size=(pos_emb, 3)).to(device) * 10
xyz_table = get_coord_table(coord_shape=reso,device=device).reshape(-1,3)
xyz_table = xyz_table[mask,:]
input_channels = pos_emb*2 + 27 + 1

CodebookNetSH = make_network(
    args.layer_sh, 
    args.channel_sh, 
    input_channels=input_channels,
    output_channels=27).to(device)
print(CodebookNetSH)

state_dict = torch.load(os.path.join(save_dir, 'CodebookNet.pth'),map_location=device)

CodebookNetSH.load_state_dict(state_dict['codebook_sh'])
CodebookNetDensity.load_state_dict(state_dict['codebook_density'])
B_density = state_dict['B_density']
B_sh = state_dict['B_sh']

with torch.no_grad():
    density_pred_all = []
    for sample_point in torch.arange(0,xyz_table.shape[0]).to(device).chunk(100):
        xyz_sampled = xyz_table[sample_point]
        xyz_sampled_fourier = input_mapping(xyz_sampled.detach().clone(),B_density)
        d_sampled = d_input[sample_point,:]
        sh_sampled = sh_input[sample_point,:]
        if args.use_lowres:
            d_lowres_sampled = d_lowres[sample_point,:]
            sh_lowres_sampled = sh_lowres[sample_point,:]
            pred = CodebookNetDensity(torch.cat([xyz_sampled_fourier, d_sampled, sh_sampled, d_lowres_sampled, sh_lowres_sampled], -1))
        else:
            pred = CodebookNetDensity(torch.cat([xyz_sampled_fourier, d_sampled, sh_sampled], -1))
        pred = pred + d_sampled
        density_pred_all.append(pred)    
    density_pred_all = torch.cat(density_pred_all,0)

with torch.no_grad():
    sh_pred_all = []
    for sample_point in torch.arange(0,xyz_table.shape[0]).to(device).chunk(100):
        xyz_sampled = xyz_table[sample_point]
        xyz_sampled_fourier = input_mapping(xyz_sampled.detach().clone(),B_sh)
        d_sampled = d_input[sample_point,:]
        sh_sampled = sh_input[sample_point,:]
        if args.use_lowres:
            d_lowres_sampled = d_lowres[sample_point,:]
            sh_lowres_sampled = sh_lowres[sample_point,:]
            pred = CodebookNetSH(torch.cat([xyz_sampled_fourier, d_sampled, sh_sampled, d_lowres_sampled, sh_lowres_sampled], -1))
        else:
            pred = CodebookNetSH(torch.cat([xyz_sampled_fourier, d_sampled, sh_sampled], -1))
        sh_pred_all.append(pred)    
    sh_pred_all = torch.cat(sh_pred_all,0)



if args.use_importance_threshold:
    total_elements = importance_map.numel()
    val, _ = torch.kthvalue(importance_map.reshape(-1),k=total_elements-int(args.threshold_percent * total_elements))
    top5_thres = val.item()

need_mlp_mask = importance_map.reshape(-1) < top5_thres

sh_full = sh_target.clone()
d_full = d_target.clone()

sh_full[need_mlp_mask,:] = sh_pred_all[need_mlp_mask,:]
d_full[need_mlp_mask,:] = density_pred_all[need_mlp_mask,:]

sh_full = (sh_full*sh_std) + sh_mean
d_full = (d_full*d_std) + d_mean


grid_input.sh_data = nn.Parameter(sh_full)
grid_input.density_data = nn.Parameter(d_full)

grid_input.density_rms= torch.zeros(1)
grid_input.sh_rms= torch.zeros(1)

grid_input.save(save_file)

torch.save({
        "codebook_sh":CodebookNetSH.state_dict(),
        "codebook_density":CodebookNetDensity.state_dict(),
        "B_sh":B_sh,
        "B_density":B_density,
        }, os.path.join(save_dir, 'CodebookNet.pth'))

os.system(f"python eval.py  {save_file}  {args.data_dir} -c {args.config}")
