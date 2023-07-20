import os
import numpy as np
import math
import torch
from torch import nn
from tqdm import tqdm
import svox2
from svox2 import utils
import argparse
from torch.nn import functional as F

_C = utils._get_c_extension()

class Vexol_Gen:

    def __init__(self, root_path, ckpt_name, device, factor=2):
        _, path_file = os.path.split(ckpt_name)
        self.path_root = root_path
        self.ckpt_path_hr = os.path.join(root_path, ckpt_name + '_hr.npz')
        self.factor = factor
        self.ckpt_path_lr_2x = os.path.join(root_path, ckpt_name + '_lr.npz')
        self.ckpt_path_lr_4x = os.path.join(root_path, ckpt_name + '_lr4x.npz')

        self.file_name = os.path.splitext(path_file)[0]
        self.crop_path_hr = os.path.join(
            self.path_root,
            'hr_sparse_ckpt_{}_{}xlr'.format(self.file_name, self.factor))
        self.crop_path_lr_2x = os.path.join(
            self.path_root, 'lr_sparse_ckpt_{}_2xlr'.format(self.file_name))
        self.crop_path_lr_4x = os.path.join(
            self.path_root, 'lr_sparse_ckpt_{}_4xlr'.format(self.file_name))

        os.makedirs(self.crop_path_hr, exist_ok=True)
        os.makedirs(self.crop_path_lr_2x, exist_ok=True)
        os.makedirs(self.crop_path_lr_4x, exist_ok=True)
        os.makedirs(os.path.join(self.path_root, 'res_ckpt'), exist_ok=True)

        self.device = device

        self.reso = np.load(self.ckpt_path_hr)
        self.reso = self.reso.f.links.shape

        self.center = np.array([0, 0, 0])
        self.radius = np.array([1])
        self._offset = torch.from_numpy(
            0.5 * (1.0 - self.center / self.radius)).float()
        self._scaling = torch.from_numpy(0.5 / self.radius).float()

        self.weight_thresh = 0.0005

        self.hr_sparse_ckpt_path = self.ckpt_path_hr.replace('dense', 'sparse')
        self.lr_sparse_ckpt_path_2x = self.ckpt_path_lr_2x.replace(
            'dense', 'sparse')
        self.lr_sparse_ckpt_path_4x = self.ckpt_path_lr_4x.replace(
            'dense', 'sparse')
        self.hr_dense_ckpt_bg = os.path.join(self.path_root,
                                             ckpt_name + '_bg.npz')
        self.index_ckpt_path = os.path.join(self.path_root,
                                            ckpt_name + '_index.npz')

        print('voxel processer has init...')

    def load_sparse(self):
        z = np.load(self.ckpt_path_hr)
        if "data" in z.keys():
            # Compatibility
            all_data = z.f.data
            sh_data = all_data[..., 1:]
            density_data = all_data[..., :1]
        else:
            sh_data = z.f.sh_data
            density_data = z.f.density_data
        return {'sh_data': sh_data, 'den_data': density_data}

    def to_sparse(self, grid, device):
        with torch.no_grad():
            mask = grid.links.view(-1) >= 0
            full_density = torch.zeros(mask.shape[0], 1).float().to(device)
            full_density[mask] = grid.density_data
            full_density = full_density.view(*grid.links.shape, 1)

            full_sh = torch.zeros(mask.shape[0], 27).float().to(device)
            full_sh[mask] = grid.sh_data
            full_sh = full_sh.view(*grid.links.shape, 27)

            print('density', full_density.shape, full_density.dtype)
            print('sh', full_sh.shape, full_sh.dtype)
            return {
                "full_density": full_density.numpy(),
                "full_sh": full_sh.numpy().astype(np.float16),
                "mask": mask
            }

    def resample_pt(self, grid_name, factor):
        grid = np.load(grid_name)

        mask = torch.from_numpy(grid.f.links).view(-1) >= 0
        full_density = torch.zeros(mask.shape[0], 1).float()
        full_density[mask] = torch.from_numpy(grid.f.density_data)
        grid_den = full_density.view(*grid.f.links.shape, 1)

        full_sh = torch.zeros(mask.shape[0], 27).float()
        full_sh[mask] = torch.from_numpy(grid.f.sh_data).float()
        grid_sh = full_sh.view(*grid.f.links.shape, 27)

        if len(grid_den.shape) == 3:
            grid_den = grid_den.unsqueeze(-1)

        grid_den = grid_den.permute(3, 2, 0, 1).unsqueeze(0)
        grid_sh = grid_sh.permute(3, 2, 0, 1).unsqueeze(0)

        lr_den = F.interpolate(grid_den,
                               scale_factor=1 / factor,
                               mode='trilinear')
        lr_sh = F.interpolate(grid_sh.float(),
                              scale_factor=1 / factor,
                              mode='trilinear')

        inter_data = {
            'full_density':
            lr_den.permute(0, 3, 4, 2, 1).squeeze().numpy(),
            'full_sh':
            lr_sh.permute(0, 3, 4, 2, 1).squeeze().numpy().astype(np.float16)
        }
        return inter_data

    def resample(
        self,
        grid,
        reso,
        sigma_thresh: float = 5.0,
        weight_thresh: float = 0.01,
        dilate: int = 2,
        use_z_order: bool = False,
        accelerate: bool = True,
        weight_render_stop_thresh:
        float = 0.2,  # SHOOT, forgot to turn this off for main exps..
        max_elements: int = 0):
        with torch.no_grad():
            device = grid.links.device
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert (
                    len(reso) == 3
                ), "reso must be an integer or indexable object of 3 ints"

            curr_reso = grid.links.shape
            dtype = torch.float32
            reso_facts = [0.5 * curr_reso[i] / reso[i] for i in range(3)]
            X = torch.linspace(
                reso_facts[0] - 0.5,
                curr_reso[0] - reso_facts[0] - 0.5,
                reso[0],
                dtype=dtype,
            )
            Y = torch.linspace(
                reso_facts[1] - 0.5,
                curr_reso[1] - reso_facts[1] - 0.5,
                reso[1],
                dtype=dtype,
            )
            Z = torch.linspace(
                reso_facts[2] - 0.5,
                curr_reso[2] - reso_facts[2] - 0.5,
                reso[2],
                dtype=dtype,
            )
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

            points = points.to(device=device)

            batch_size = 720720
            all_sample_vals_density = []
            all_sample_vals_sh = []
            print('Pass 1/2 (density)')
            for i in tqdm(range(0, len(points), batch_size)):
                sample_vals_density, sample_vals_sh = grid.sample(
                    points[i:i + batch_size],
                    grid_coords=True,
                    use_kernel=False,
                    want_colors=True)
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)
                all_sample_vals_sh.append(sample_vals_sh)
            grid.density_data.grad = None
            grid.sh_data.grad = None
            grid.sparse_grad_indexer = None
            grid.sparse_sh_grad_indexer = None
            grid.density_rms = None
            grid.sh_rms = None

            sample_vals_density = torch.cat(all_sample_vals_density,
                                            dim=0).view(reso)
            sample_vals_sh = torch.cat(all_sample_vals_sh, dim=0).view(
                (reso[0], reso[1], reso[2], 27))
            del all_sample_vals_density
            del all_sample_vals_sh

            print('density', sample_vals_density.shape,
                  sample_vals_density.dtype)
            print('sh', sample_vals_sh.shape, sample_vals_sh.dtype)
            density_data = sample_vals_density.numpy()
            sh_data = sample_vals_sh.numpy().astype(np.float16)
            return {"full_density": density_data, "full_sh": sh_data}

    def make_data_sparse(self, use_sample=False, use_bg=False):
        grid_hr = svox2.SparseGrid.load(self.ckpt_path_hr, device='cpu')
        if use_bg == 'sfTnTBG':
            grid_bg_dense = self.to_bg_dense(grid_hr, device='cpu')
            np.savez(self.hr_dense_ckpt_bg, **grid_bg_dense)

        if use_sample:
            grid_lr = self.resample(
                grid_hr,
                [self.reso[0] // 2, self.reso[1] // 2, self.reso[2] // 2],
                sigma_thresh=5.0,
                weight_thresh=0.0,
                dilate=0,  # use_sparsify,
                max_elements=44_000_000)
            np.savez(self.lr_sparse_ckpt_path_2x, **grid_lr)
            grid_lr = self.resample(
                grid_hr,
                [self.reso[0] // 4, self.reso[1] // 4, self.reso[2] // 4],
                sigma_thresh=5.0,
                weight_thresh=0.0,
                dilate=0,  # use_sparsify,
                max_elements=44_000_000)
            np.savez(self.lr_sparse_ckpt_path_4x, **grid_lr)
        else:
            grid_lr = svox2.SparseGrid.load(self.ckpt_path_lr_2x, device='cpu')
            grid_lr_sparse = self.to_sparse(grid_lr, device='cpu')
            np.savez(self.lr_sparse_ckpt_path_2x, **grid_lr_sparse)
            grid_lr = svox2.SparseGrid.load(self.ckpt_path_lr_4x, device='cpu')
            grid_lr_sparse = self.to_sparse(grid_lr, device='cpu')
            np.savez(self.lr_sparse_ckpt_path_4x, **grid_lr_sparse)

        grid_index = np.zeros_like(grid_hr.links).astype(bool)
        grid_index[grid_hr.links >= 0] = True
        dic_index = {'dic_index': grid_index}
        grid_hr_sparse = self.to_sparse(grid_hr, device='cpu')

        np.savez(self.hr_sparse_ckpt_path, **grid_hr_sparse)
        np.savez(self.index_ckpt_path, **dic_index)

    def compute_dense(self,
                      grid_name,
                      use_weight_thresh,
                      dilate=2,
                      sigma_thresh=5.0,
                      from_gpu=False,
                      max_elements=44_000_000,
                      dataroot=''):
        if from_gpu:
            grid_src = grid_name
            sample_vals_density = grid_src['full_density'].squeeze()
            sample_vals_sh = grid_src['full_sh']
            reso = sample_vals_density.shape
        else:
            if isinstance(grid_name, str):
                grid_src = np.load(os.path.join(self.path_root, grid_name))
                d_src = grid_src.f.full_density.squeeze()
                d_sh = grid_src.f.full_sh
            else:
                grid_src = grid_name
                d_src = grid_src['full_density'].squeeze()
                d_sh = grid_src['full_sh']

            reso = d_src.shape
            sample_vals_density = torch.from_numpy(d_src).contiguous().to(
                self.device)
            sample_vals_sh = torch.from_numpy(d_sh).contiguous().to(
                self.device)

        if use_weight_thresh == 'weight':
            from opt import opt
            self.cameras = opt.resample_cameras

            gsz = torch.tensor(reso)
            offset = (self._offset * gsz - 0.5).to(device=self.device)
            scaling = (self._scaling * gsz).to(device=self.device)
            max_wt_grid = torch.zeros(reso,
                                      dtype=torch.float32,
                                      device=self.device)
            print(" Grid weight render", sample_vals_density.shape)
            for i, cam in enumerate(self.cameras):
                _C.grid_weight_render(
                    sample_vals_density,
                    cam._to_cpp(),
                    0.5,
                    0.2,
                    #  self.opt.last_sample_opaque,
                    False,
                    offset,
                    scaling,
                    max_wt_grid)
            sample_vals_mask = max_wt_grid >= self.weight_thresh
            if max_elements > 0 and max_elements < max_wt_grid.numel() \
                                and max_elements < torch.count_nonzero(sample_vals_mask):
                # To bound the memory usage
                weight_thresh_bounded = torch.topk(
                    max_wt_grid.view(-1), k=max_elements,
                    sorted=False).values.min().item()
                weight_thresh = max(self.weight_thresh, weight_thresh_bounded)
                print(' Readjusted weight thresh to fit to memory:',
                      weight_thresh)
                sample_vals_mask = max_wt_grid >= weight_thresh
            del max_wt_grid
        elif use_weight_thresh == 'mask':
            sample_vals_mask = np.load(self.index_ckpt_path)
            sample_vals_mask = torch.tensor(
                sample_vals_mask.f.dic_index.astype(bool))
        elif use_weight_thresh == 'sigma':
            sample_vals_mask = sample_vals_density >= sigma_thresh
            print(' adjust density with sigma:', sigma_thresh)

        if dilate:
            for i in range(int(dilate)):
                sample_vals_mask = _C.dilate(sample_vals_mask)
        sample_vals_mask = sample_vals_mask.view(-1)
        sample_vals_density = sample_vals_density.view(-1)
        sample_vals_density = sample_vals_density[sample_vals_mask]
        sample_vals_density = sample_vals_density.view(-1, 1)

        sample_vals_sh = sample_vals_sh.view(-1, 27)
        sample_vals_sh = sample_vals_sh.to('cpu')[
            sample_vals_mask.to('cpu'), :]
        init_links = (
            torch.cumsum(sample_vals_mask.to(torch.int32), dim=-1).int() - 1)
        init_links[~sample_vals_mask] = -1
        init_links = init_links.view(reso).to(device=self.device)

        assert (_C is not None and init_links.is_cuda
                ), "CUDA extension is currently required for accelerate"
        _C.accel_dist_prop(init_links)

        if dataroot == 'MEET':
            res_radius = np.array([1.390625, 1.6944444, 1.0])
        elif dataroot == 'LLFF':
            res_radius = np.array([1.4960318, 1.6613756, 1.0])
        elif dataroot == 'SYN' or 'NSVF':
            res_radius = np.array([1, 1, 1])
        ret = {
            "radius": res_radius,
            "center": self.center,
            "links": init_links.cpu().numpy(),
            "density_data": sample_vals_density.cpu().numpy(),
            "sh_data": sample_vals_sh.cpu().numpy().astype(np.float16),
            "basis_type": 1
        }
        return ret

    def load_crop(self):
        hr_list = os.listdir(self.crop_path_hr)
        lr_list = os.listdir(self.crop_path_lr)
        for item in hr_list:
            v_crop = np.load(os.path.join(self.crop_path_hr, item))
            v_d = v_crop.f.density
            v_sh = v_crop.f.sh

    def voxel_embeding(self, factor):
        X = torch.arange(self.reso[0] // factor,
                         dtype=torch.int16,
                         device='cpu')
        Y = torch.arange(self.reso[1] // factor,
                         dtype=torch.int16,
                         device='cpu')
        Z = torch.arange(self.reso[2] // factor,
                         dtype=torch.int16,
                         device='cpu')
        X, Y, Z = torch.meshgrid(X, Y, Z)
        points = torch.stack((X, Y, Z), dim=-1)
        embed_fn, input_ch = get_embedder(10)
        ebed_p = embed_fn(points)
        return ebed_p

    def sample_grid_new(self,
                        crop_size_hr=128,
                        step_hr=64,
                        thresh_size=0,
                        use_pe=False,
                        use_gpu=False):
        grid_gt = np.load(self.hr_sparse_ckpt_path)
        grid_lr_2x = np.load(self.lr_sparse_ckpt_path_2x)
        grid_lr_4x = np.load(self.lr_sparse_ckpt_path_4x)
        hhr, whr, zhr = grid_gt.f.full_density.shape[0:3]
        grid_index = np.load(self.index_ckpt_path)
        if use_gpu:
            # grid_gt_sh_pt = torch.zeros((hhr, whr, zhr, 27), dtype=torch.half)
            # grid_gt_d_pt = torch.zeros((hhr, whr, zhr, 1))
            grid_gt_sh_pt = torch.from_numpy(grid_gt.f.full_sh)
            grid_gt_d_pt = torch.from_numpy(grid_gt.f.full_density).squeeze()
            grid_gt_pt = [
                grid_gt_sh_pt.to(self.device),
                grid_gt_d_pt.to(self.device)
            ]

            # grid_lr_2x_pt = torch.zeros((hhr // 2, whr // 2, zhr // 2, 28))
            grid_lr_2x_sh_pt = torch.from_numpy(grid_lr_2x.f.full_sh)
            grid_lr_2x_d_pt = torch.from_numpy(
                grid_lr_2x.f.full_density).squeeze()
            grid_lr_2x_pt = [
                grid_lr_2x_sh_pt.to(self.device),
                grid_lr_2x_d_pt.to(self.device)
            ]

            # grid_lr_4x_pt = torch.zeros((hhr // 4, whr // 4, zhr // 4, 28))
            grid_lr_4x_sh_pt = torch.from_numpy(grid_lr_4x.f.full_sh)
            grid_lr_4x_d_pt = torch.from_numpy(
                grid_lr_4x.f.full_density).squeeze()
            grid_lr_4x_pt = [
                grid_lr_4x_sh_pt.to(self.device),
                grid_lr_4x_d_pt.to(self.device)
            ]

            grid_index = torch.from_numpy(grid_index.f.dic_index).to(
                self.device)

        crop_size_lr_2x = crop_size_hr // 2
        crop_size_lr_4x = crop_size_hr // 4

        thre_empty = crop_size_hr * crop_size_hr * crop_size_hr / 64
        thre_tiny = crop_size_hr * crop_size_hr * crop_size_hr / 16

        hhr_space = np.arange(0, hhr - crop_size_hr + 1, step_hr)
        if hhr - (hhr_space[-1] + crop_size_hr) > thresh_size:
            hhr_space = np.append(hhr_space, hhr - crop_size_hr)
        whr_space = np.arange(0, whr - crop_size_hr + 1, step_hr)
        if whr - (whr_space[-1] + crop_size_hr) > thresh_size:
            whr_space = np.append(whr_space, whr - crop_size_hr)
        zhr_space = np.arange(0, zhr - crop_size_hr + 1, step_hr)
        if zhr - (zhr_space[-1] + crop_size_hr) > thresh_size:
            zhr_space = np.append(zhr_space, zhr - crop_size_hr)

        if use_pe:
            pe_lr_2x = self.voxel_embeding(2)
            pe_lr_4x = self.voxel_embeding(4)

        index = 0
        for x in hhr_space:
            for y in whr_space:
                for z in zhr_space:
                    if use_gpu:
                        cropped_index = grid_index[x:x + crop_size_hr,
                                                   y:y + crop_size_hr,
                                                   z:z + crop_size_hr]
                        trigger = np.random.random()
                        if (torch.sum(cropped_index) < thre_empty):
                            print('smaller than 1/64, jumped.')
                            continue
                        elif (torch.sum(cropped_index) <
                              thre_tiny) and (trigger > 0.4):
                            print(
                                'bigger than 1/64 and smaller than 1/16, jumped.'
                            )
                            continue

                        cropped_grid_gt_sh = grid_gt_pt[0][x:x + crop_size_hr,
                                                           y:y + crop_size_hr,
                                                           z:z +
                                                           crop_size_hr, :]
                        cropped_grid_gt_den = grid_gt_pt[1][x:x + crop_size_hr,
                                                            y:y + crop_size_hr,
                                                            z:z + crop_size_hr]
                        if torch.max(cropped_grid_gt_den) < 2:
                            print('max sh samller than 2, jumped.')
                            continue
                        cropped_grid_gt_sh = cropped_grid_gt_sh.cpu().numpy()
                        cropped_grid_gt_den = cropped_grid_gt_den.cpu().numpy()

                        lx_2x = x // 2
                        ly_2x = y // 2
                        lz_2x = z // 2
                        cropped_grid_lr2x_sh = grid_lr_2x_pt[0][
                            lx_2x:lx_2x + crop_size_lr_2x,
                            ly_2x:ly_2x + crop_size_lr_2x,
                            lz_2x:lz_2x + crop_size_lr_2x, :]
                        cropped_grid_lr2x_den = grid_lr_2x_pt[1][
                            lx_2x:lx_2x + crop_size_lr_2x,
                            ly_2x:ly_2x + crop_size_lr_2x,
                            lz_2x:lz_2x + crop_size_lr_2x]
                        cropped_grid_lr2x_sh = cropped_grid_lr2x_sh.cpu(
                        ).numpy()
                        cropped_grid_lr2x_den = cropped_grid_lr2x_den.cpu(
                        ).numpy()

                        lx_4x = x // 4
                        ly_4x = y // 4
                        lz_4x = z // 4
                        cropped_grid_lr4x_sh = grid_lr_4x_pt[0][
                            lx_4x:lx_4x + crop_size_lr_4x,
                            ly_4x:ly_4x + crop_size_lr_4x,
                            lz_4x:lz_4x + crop_size_lr_4x, :]
                        cropped_grid_lr4x_den = grid_lr_4x_pt[1][
                            lx_4x:lx_4x + crop_size_lr_4x,
                            ly_4x:ly_4x + crop_size_lr_4x,
                            lz_4x:lz_4x + crop_size_lr_4x]
                        cropped_grid_lr4x_sh = cropped_grid_lr4x_sh.cpu(
                        ).numpy()
                        cropped_grid_lr4x_den = cropped_grid_lr4x_den.cpu(
                        ).numpy()
                    else:
                        cropped_index = grid_index.f.dic_index[x:x +
                                                               crop_size_hr,
                                                               y:y +
                                                               crop_size_hr,
                                                               z:z +
                                                               crop_size_hr]
                        trigger = np.random.random()
                        if (np.sum(cropped_index) < thre_empty):
                            print('smaller than 1/64, jumped.')
                            continue
                        elif (np.sum(cropped_index) < thre_tiny) and (trigger >
                                                                      0.4):
                            print(
                                'bigger than 1/64 and smaller than 1/16, jumped.'
                            )
                            continue

                        cropped_grid_gt_den = grid_gt.f.full_density[
                            x:x + crop_size_hr, y:y + crop_size_hr,
                            z:z + crop_size_hr, ...]
                        cropped_grid_gt_den = np.ascontiguousarray(
                            cropped_grid_gt_den)
                        cropped_grid_gt_sh = grid_gt.f.full_sh[x:x +
                                                               crop_size_hr,
                                                               y:y +
                                                               crop_size_hr,
                                                               z:z +
                                                               crop_size_hr,
                                                               ...]
                        cropped_grid_gt_sh = np.ascontiguousarray(
                            cropped_grid_gt_sh)
                        if np.max(cropped_grid_gt_den) < 2:
                            print('max sh equal to min sh, jumped.')
                            continue

                        lx_2x = x // 2
                        ly_2x = y // 2
                        lz_2x = z // 2
                        cropped_grid_lr2x_den = grid_lr_2x.f.full_density[
                            lx_2x:lx_2x + crop_size_lr_2x,
                            ly_2x:ly_2x + crop_size_lr_2x,
                            lz_2x:lz_2x + crop_size_lr_2x, ...]
                        cropped_grid_lr2x_den = np.ascontiguousarray(
                            cropped_grid_lr2x_den)
                        cropped_grid_lr2x_sh = grid_lr_2x.f.full_sh[
                            lx_2x:lx_2x + crop_size_lr_2x,
                            ly_2x:ly_2x + crop_size_lr_2x,
                            lz_2x:lz_2x + crop_size_lr_2x, ...]
                        cropped_grid_lr2x_sh = np.ascontiguousarray(
                            cropped_grid_lr2x_sh)
                        cropped_grid_lr2x = {
                            "density": cropped_grid_lr2x_den,
                            "sh": cropped_grid_lr2x_sh
                        }

                        lx_4x = x // 4
                        ly_4x = y // 4
                        lz_4x = z // 4
                        cropped_grid_lr4x_den = grid_lr_4x.f.full_density[
                            lx_4x:lx_4x + crop_size_lr_4x,
                            ly_4x:ly_4x + crop_size_lr_4x,
                            lz_4x:lz_4x + crop_size_lr_4x, ...]
                        cropped_grid_lr4x_den = np.ascontiguousarray(
                            cropped_grid_lr4x_den)
                        cropped_grid_lr4x_sh = grid_lr_4x.f.full_sh[
                            lx_4x:lx_4x + crop_size_lr_4x,
                            ly_4x:ly_4x + crop_size_lr_4x,
                            lz_4x:lz_4x + crop_size_lr_4x, ...]
                        cropped_grid_lr4x_sh = np.ascontiguousarray(
                            cropped_grid_lr4x_sh)

                    cropped_grid_gt = {
                        "density": cropped_grid_gt_den,
                        "sh": cropped_grid_gt_sh
                    }
                    cropped_grid_lr2x = {
                        "density": cropped_grid_lr2x_den,
                        "sh": cropped_grid_lr2x_sh
                    }
                    cropped_grid_lr4x = {
                        "density": cropped_grid_lr4x_den,
                        "sh": cropped_grid_lr4x_sh
                    }

                    index += 1
                    np.savez(
                        os.path.join(
                            self.crop_path_hr,
                            self.file_name + '_hr{:0>4d}.npz'.format(index)),
                        **cropped_grid_gt)
                    np.savez(
                        os.path.join(
                            self.crop_path_lr_2x,
                            self.file_name + '_lr{:0>4d}.npz'.format(index)),
                        **cropped_grid_lr2x)
                    np.savez(
                        os.path.join(
                            self.crop_path_lr_4x,
                            self.file_name + '_lr{:0>4d}.npz'.format(index)),
                        **cropped_grid_lr4x)

                    print(
                        '{} vexols patch has saved. Max hr sh is {}, Max lr sh is {}. Min hr sh is {}, Min lr sh is {}'
                        .format(index, np.max(cropped_grid_gt_sh),
                                np.max(cropped_grid_lr4x_sh),
                                np.min(cropped_grid_gt_sh),
                                np.min(cropped_grid_lr4x_sh)))


def spread_dense(sh_path, den_path):
    grid_2x = np.load(den_path)
    grid_4x = np.load(sh_path)
    return [
        grid_4x.f.sh_data, grid_4x.f.density_data, grid_4x.f.links,
        grid_2x.f.sh_data, grid_2x.f.density_data, grid_2x.f.links
    ]


def spread_single(grid_path):
    grid_sp = np.load(grid_path)
    np.savez(grid_path.replace('.npz', '_links.npz'), grid_sp.f.links)


def reload_dense_den(sh_path, den_path):
    grid_den = np.load(den_path)
    grid_sh = np.load(sh_path)
    return {
        "density_data": grid_den.f.density_data,
        "sh_data": grid_sh.f.sh_data,
        "links": grid_den.f.links
    }


def thresh_mask(vx_path, thre_den):
    grid_mask = np.load(vx_path)
    sample_vals_mask = grid_mask.f.density_data >= thre_den
    sample_vals_mask = np.squeeze(sample_vals_mask)
    sample_vals_density = grid_mask.f.density_data
    sample_vals_density = sample_vals_density[sample_vals_mask]
    sample_vals_sh = grid_mask.f.sh_data
    sample_vals_sh = sample_vals_sh[sample_vals_mask, :]
    cnz = np.sum(sample_vals_mask).item()
    init_links = (np.cumsum(sample_vals_mask.astype(np.int32), axis=-1) -
                  1).astype(np.int32)
    init_links[~sample_vals_mask] = -1
    links = np.reshape(init_links, grid_mask.f.links.shape)
    print(" New cap:", cnz)
    return {
        "radius": grid_mask.f.radius,
        "center": grid_mask.f.center,
        "links": links,
        "density_data": sample_vals_density,
        "sh_data": sample_vals_sh.astype(np.float16),
        "basis_type": 1
    }


def reload_vxdata(vx_path, dst_path):
    vx_list = os.listdir(vx_path)
    os.makedirs(dst_path, exist_ok=True)

    for vx_p in tqdm(vx_list):
        vx_f = os.path.join(vx_path, vx_p)

        grid_data = np.load(vx_f)
        grid_data = {
            "density": grid_data.f.density,
            "sh": grid_data.f.sh[:, :, :, :27],
        }
        np.savez(os.path.join(dst_path, vx_p), **grid_data)


def bitfy_link(link_path, save_path=None):
    if not save_path:
        sample_vals_mask = np.load(link_path)
        sample_vals_mask = sample_vals_mask.f.dic_index
        bit_mask = np.packbits(sample_vals_mask)
        np.savez(link_path.replace('index', 'bitlink'), bit_mask)
        cmd_line = 'zip {0} {1}'.format(link_path.replace('.npz', '.zip'),
                                        link_path.replace('index', 'bitlink'))
        os.system(cmd_line)
        # cmd_line = 'rm -f {}'.format(link_path.replace('index', 'bitlink'))
        # os.system(cmd_line)
    else:
        sample_vals_mask = np.zeros_like(link_path).astype(np.bool)
        sample_vals_mask[link_path >=0] = True
        bit_mask = np.packbits(sample_vals_mask)
        np.savez(save_path, bit_mask)
        cmd_line = 'zip {0} {1}'.format(save_path.replace('.npz', '.zip'),
                                        save_path.replace('link', 'bitlink'))
        os.system(cmd_line)
        # cmd_line = 'rm -f {}'.format(save_path)
        # os.system(cmd_line)


def main(data_name, data_root, mode, vx_src):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',
                        type=str,
                        default='lego',
                        help='used dataset names')
    parser.add_argument('--mode',
                        type=str,
                        default='make_sparse',
                        help='mode to run')
    parser.add_argument('--data_root',
                        type=str,
                        default='SYN',
                        help='dataset root dir')
    parser.add_argument('--vx_src',
                        type=str,
                        default='',
                        help='mode to run')
    args = parser.parse_args()

    data_name = args.data_name
    data_root = args.data_root
    mode = args.mode

    vexgen = Vexol_Gen('/mnt/disk1/cuixiao/res_vox/{0}/{1}'.format(
        data_root, data_name),
                       'ckpt_src/ckpt_{}_dense'.format(data_name),
                       device='cuda:0',
                       factor=4)

    if mode == 'recover_dense':
        vexgen.make_data_sparse(use_sample=True, use_bg=data_root)
    elif mode == 'make_sr':
        vexgen.sample_grid_new(crop_size_hr=72,
                               step_hr=36,
                               use_pe=False,
                               use_index=False,
                               use_gpu=True)
    elif mode == 'densify':
        res = vexgen.compute_dense('res_ckpt/{}'.format(vx_src),
                                   use_weight_thresh='weight',
                                   dilate=0,
                                   dataroot=data_root)
        save_path = os.path.join(
            vexgen.path_root,
            'res_ckpt/{}'.format(vx_src.replace('sparse', 'dense_wm_')))
        np.savez(save_path, **res)
    elif mode == 'spread_lr':
        res = spread_dense(
            os.path.join(vexgen.path_root,
                         'ckpt_src/ckpt_{}_dense_lr4x.npz'.format(data_name)),
            os.path.join(vexgen.path_root,
                         'ckpt_src/ckpt_{}_dense_lr.npz'.format(data_name)))
        os.makedirs(os.path.join(vexgen.path_root, 'res_ckpt'), exist_ok=True)
        np.savez(
            os.path.join(vexgen.path_root,
                         'res_ckpt/ckpt_{0}_4xsh.npz'.format(data_name)),
            res[0])
        np.savez(
            os.path.join(vexgen.path_root,
                         'res_ckpt/ckpt_{0}_4xd.npz'.format(data_name)),
            res[1])
        bitfy_link(
            res[2],
            os.path.join(vexgen.path_root,
                         'res_ckpt/ckpt_{0}_4x_index.npz'.format(data_name)))
        np.savez(
            os.path.join(vexgen.path_root,
                         'res_ckpt/ckpt_{0}_2xsh.npz'.format(data_name)),
            res[3])
        np.savez(
            os.path.join(vexgen.path_root,
                         'res_ckpt/ckpt_{0}_2xd.npz'.format(data_name)),
            res[4])
        bitfy_link(
            res[5],
            os.path.join(vexgen.path_root,
                         'res_ckpt/ckpt_{0}_2x_index.npz'.format(data_name)))

    elif mode == 'densify_lr':
        grid_data = vexgen.compute_dense(
            'ckpt_src/ckpt_{0}_sparse_lr4x.npz'.format(data_name),
            use_weight_thresh='weight',
            dilate=2,
            dataroot=data_root)
        np.savez(
            os.path.join(vexgen.path_root,
                         'ckpt_src/ckpt_{0}_dense_lr4x.npz'.format(data_name)),
            **grid_data)
        grid_data = vexgen.compute_dense(
            'ckpt_src/ckpt_{0}_sparse_lr.npz'.format(data_name),
            use_weight_thresh='weight',
            dilate=2,
            dataroot=data_root)
        np.savez(
            os.path.join(vexgen.path_root,
                         'ckpt_src/ckpt_{0}_dense_lr.npz'.format(data_name)),
            **grid_data)
    elif mode == 'bitfy_link':
        bitfy_link(
            os.path.join(
                vexgen.path_root,
                'ckpt_src/ckpt_{0}_dense_index.npz'.format(data_name)))
    elif mode == 'tri':
        grid_data_den = vexgen.resample_pt(
            os.path.join(vexgen.path_root,
                         'ckpt_src/ckpt_{0}_dense_lr.npz'.format(data_name)),
            1 / 2)
        grid_data_sh = vexgen.resample_pt(
            os.path.join(
                vexgen.path_root,
                'ckpt_src/ckpt_{0}_dense_lr4x.npz'.format(data_name)), 1 / 4)
        grid_data = {
            'full_sh': grid_data_sh['full_sh'],
            'full_density': grid_data_den['full_density']
        }
        grid_data = vexgen.compute_dense(grid_data,
                                         use_weight_thresh='mask',
                                         dilate=0,
                                         from_gpu=False,
                                         dataroot=data_root)
        np.savez(
            os.path.join(
                vexgen.path_root,
                'res_ckpt/ckpt_{0}_dense_2xd4xsh.npz'.format(data_name)),
            **grid_data)


if __name__ == '__main__':
    # data_name = 'lego'
    # data_root = 'SYN'
    # mode = 'spread_hr'
    # vx_src = 'ckpt_lego_sparse_lr_synlego_dnr_plane_ps_24B64P_l1.npz'
    main()
