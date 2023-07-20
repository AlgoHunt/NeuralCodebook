import argparse
import svox2
import torch
import os
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.cuda import amp
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


from math import sqrt
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm1d(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        # print(style.shape)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        # print(out.shape)
        out = gamma * out + beta

        return out



    
class cross_attention_net(nn.Module):
    def __init__(self, sh_channels, pos_channels, num_channels, output_channels):
        super().__init__()
        self.sh_channels = sh_channels
        self.pos_channels = pos_channels
        

        self.mlp1 = nn.Sequential(*[
            nn.Linear(28, num_channels),
            nn.LeakyReLU()
        ])

        self.adain1 = AdaptiveInstanceNorm1d(num_channels, pos_channels)
        self.mlp2 = nn.Sequential(*[
            nn.Linear(num_channels,num_channels),
            nn.LeakyReLU()
        ])
        self.adain2 = AdaptiveInstanceNorm1d(num_channels, pos_channels)
        self.mlp3 = nn.Sequential(*[
            nn.Linear(num_channels,num_channels),
            nn.LeakyReLU()
        ])
        self.adain3 = AdaptiveInstanceNorm1d(num_channels, pos_channels)
        self.mlp4 = nn.Sequential(*[
            nn.Linear(num_channels,num_channels),
            nn.LeakyReLU()
        ])
        self.adain4 = AdaptiveInstanceNorm1d(num_channels, pos_channels)

        self.mlp_sh_out_1 = nn.Sequential(*[
            nn.Linear(num_channels,num_channels),
            nn.LeakyReLU(),
        ])
        self.adain_sh_out_1 = AdaptiveInstanceNorm1d(num_channels, pos_channels)
        self.mlp_sh_out_2 = nn.Sequential(*[
            nn.Linear(num_channels,num_channels),
            nn.LeakyReLU(),
        ])
        self.adain_sh_out_2 = AdaptiveInstanceNorm1d(num_channels, pos_channels)

        self.mlp_sh_out_3 = nn.Sequential(*[
            nn.Linear(num_channels,num_channels),
            nn.LeakyReLU(),
        ])
        self.adain_sh_out_3 = AdaptiveInstanceNorm1d(num_channels, pos_channels)

        self.mlp_sh_out_4 = nn.Sequential(*[
            nn.Linear(num_channels,27),
        ])


        self.mlp_d_out_1 = nn.Sequential(*[
            nn.Linear(num_channels, num_channels//2),
            nn.LeakyReLU(),
        ])
        self.adain_d_out = AdaptiveInstanceNorm1d(num_channels//2, pos_channels)
        self.mlp_d_out_2 = nn.Sequential(*[
            nn.Linear(num_channels//2, 1),
        ])


        
        
        

    def forward(self, sh, density, posemb):
       
        # sh = self.mlp_sh(sh)
        # density = self.mlp_d(density)
        x = torch.cat([sh,density], 1)
        x = self.mlp1(x)
        x = self.adain1(input=x,style=posemb)
        x = self.mlp2(x)
        x = self.adain2(input=x,style=posemb)
        x = self.mlp3(x)
        x = self.adain3(input=x,style=posemb)
        x = self.mlp4(x)
        x = self.adain4(input=x,style=posemb)

        sh = self.mlp_sh_out_1(x)
        sh = self.adain_sh_out_1(input=sh, style=posemb)
        sh = self.mlp_sh_out_2(sh)
        sh = self.adain_sh_out_2(input=sh, style=posemb)
        sh = self.mlp_sh_out_3(sh)
        sh = self.adain_sh_out_3(input=sh, style=posemb)
        sh = self.mlp_sh_out_4(sh)

        # sh = self.adain_sh_out_3(input=sh, style=posemb)
        # sh = self.mlp_sh_out_4(sh)

        d = self.mlp_d_out_1(x)
        d = self.adain_d_out(input=d, style=posemb)
        d = self.mlp_d_out_2(d)

        return sh, d

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
    train.add_argument("--lr_density", type=float, default=1e-2, help="lr for density code book")
    train.add_argument("--lr_sh", type=float, default=1e-2, help="lr for sh code book")

    train.add_argument("--weight_decay_density", type=float, default=1e-5, help="weight decay for density code book")
    train.add_argument("--weight_decay_sh", type=float, default=1e-5, help="weight decay for sh code book")
    train.add_argument("--gpu", "-g" ,type=int, default=0, help="gpu num")
    train.add_argument("--use_lowres", action="store_true", help="load lr model as input too")
    train.add_argument("-c", "--config", type=str, default='configs/llff.json', help="load lr model as input too")

    train.add_argument("--use_importance_reweighting", action='store_true', help="apply loss reweighting based on importance map")
    train.add_argument("--use_importance_threshold", action='store_true', help="load lr model as input too")
    train.add_argument("--threshold_percent", type=float,default=0.05, help="apply loss reweighting based on importance map")

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
device = f"cuda:{args.gpu}"
grid_input = svox2.SparseGrid.load(args.sr_ckpt, device=device)
grid_target = svox2.SparseGrid.load(args.hr_ckpt,device=device)

mask_input = grid_input.links >= 0
mask_target = grid_target.links >= 0
assert mask_input.sum() == mask_target.sum(), f'{mask_input.sum()} vs {mask_target.sum()}, sr`s mask should equal to target`s mask'

reso = grid_input.links.shape

os.makedirs(save_dir, exist_ok=True)

if args.use_importance_reweighting or args.use_importance_threshold:
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




print("Pass 1/2 density codebook training: ")
# ---- density codebook training
pos_emb = args.pos_emb                  # 128
B_share = torch.normal(mean=0, std=1, size=(pos_emb, 3)).to(device) * 10
xyz_table = get_coord_table(coord_shape=reso,device=device).reshape(-1,3)
xyz_table = xyz_table[mask,:]


if args.use_importance_threshold:
    total_elements = importance_map.numel()
    val, _ = torch.kthvalue(importance_map.reshape(-1),k=total_elements-int(args.threshold_percent * total_elements))
    top5_thres = val.item()

    importance_threshold_mask = importance_map.reshape(-1) < top5_thres

    d_target_original = d_target
    sh_target_original = sh_target

    new_importance_map = importance_map.clone()
    new_importance_map[importance_threshold_mask] = 1

one_target = torch.cat([sh_target, d_target], -1)





input_channels = pos_emb*2 + 27 + 1


# ---- sh codebook trainingf
print("Pass 2/2 sh codebook training: ")

# CodebookNetDensity = make_network(
#     args.layer_density, 
#     args.channel_density, 
#     input_channels=input_channels ,
#     output_channels=1).to(device)

CodebookNetAll  = cross_attention_net(27,pos_emb*2,num_channels=512,output_channels=28).to(device)

state_dict = torch.load(os.path.join(save_dir, 'CodebookNet.pth'),map_location=device)
CodebookNetAll.load_state_dict(state_dict['codebook_all'])
B_share = state_dict['B_share']

with torch.no_grad():
    density_pred_all = []
    sh_pred_all = []
    for sample_point in torch.arange(0,xyz_table.shape[0]).to(device).chunk(100):
        xyz_sampled = xyz_table[sample_point]
        xyz_sampled_fourier = input_mapping(xyz_sampled.detach().clone(),B_share)
        d_sampled = d_input[sample_point,:]
        sh_sampled = sh_input[sample_point,:]

        sh,d = CodebookNetAll(sh_sampled, d_sampled, xyz_sampled_fourier)
        sh = sh_sampled + sh
        d = d_sampled + d

        density_pred_all.append(d)    
        sh_pred_all.append(sh)

    density_pred_all = torch.cat(density_pred_all,0)
    sh_pred_all = torch.cat(sh_pred_all,0)



if args.use_importance_threshold:
    sh_full = sh_target_original.clone()
    d_full = d_target_original.clone()

    sh_full[importance_threshold_mask,:] = sh_pred_all[importance_threshold_mask,:]
    d_full[importance_threshold_mask,:] = density_pred_all[importance_threshold_mask,:]

    sh_pred_all = sh_full
    density_pred_all = d_full

sh_pred_all = (sh_pred_all*sh_std) + sh_mean
density_pred_all = (density_pred_all*d_std) + d_mean


grid_input.sh_data = nn.Parameter(sh_pred_all)
grid_input.density_data = nn.Parameter(density_pred_all)

grid_input.density_rms= torch.zeros(1)
grid_input.sh_rms= torch.zeros(1)

grid_input.save(save_file)

# torch.save({
#         "codebook_all":CodebookNetAll.state_dict(),
#         "B_share":B_share,
#         }, os.path.join(save_dir, 'CodebookNet.pth'))

os.system(f"python eval.py  {save_file}  {args.data_dir} -c {args.config}")
