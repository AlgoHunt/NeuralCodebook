import argparse
import svox2
import torch
import os
import lpips
from tqdm import tqdm

from util import config_util

def define_data_args(parser : argparse.ArgumentParser):
    group = parser.add_argument_group("Data loading")
    group.add_argument('--dataset_type',
                         default="auto",
                         help="Dataset type (specify type or use auto)")
    group.add_argument('--scene_scale',
                         type=float,
                         default=None,
                         help="Global scene scaling (or use dataset default)")
    group.add_argument('--scale',
                         type=float,
                         default=None,
                         help="Image scale, e.g. 0.5 for half resolution (or use dataset default)")
    group.add_argument('--seq_id',
                         type=int,
                         default=1000,
                         help="Sequence ID (for CO3D only)")
    group.add_argument('--epoch_size',
                         type=int,
                         default=12800,
                         help="Pseudo-epoch size in term of batches (to be consistent across datasets)")
    group.add_argument('--white_bkgd',
                         type=bool,
                         default=True,
                         help="Whether to use white background (ignored in some datasets)")
    group.add_argument('--llffhold',
                         type=int,
                         default=8,
                         help="LLFF holdout every")
    group.add_argument('--normalize_by_bbox',
                         type=bool,
                         default=False,
                         help="Normalize by bounding box in bbox.txt, if available (NSVF dataset only); precedes normalize_by_camera")
    group.add_argument('--data_bbox_scale',
                         type=float,
                         default=1.2,
                         help="Data bbox scaling (NSVF dataset only)")
    group.add_argument('--cam_scale_factor',
                         type=float,
                         default=0.95,
                         help="Camera autoscale factor (NSVF/CO3D dataset only)")
    group.add_argument('--normalize_by_camera',
                         type=bool,
                         default=True,
                         help="Normalize using cameras, assuming a 360 capture (NSVF dataset only); only used if not normalize_by_bbox")
    group.add_argument('--perm', action='store_true', default=False,
                         help='sample by permutation of rays (true epoch) instead of '
                              'uniformly random rays')
    return parser.parse_args()

parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)
parser.add_argument("--gpu", "-g" ,type=int, default=0, help="gpu num")

config_util.define_common_args(parser)

args = parser.parse_args()
device = f"cuda:{args.gpu}"
config_util.maybe_merge_config_file(args, allow_invalid=True)



def get_metric(ckpt, data_dir):
    from util.dataset import datasets
    from util.util import compute_ssim
    import math
    grid = svox2.SparseGrid.load(ckpt, device=device)
    dset = datasets['auto'](data_dir, split="test",
                            **config_util.build_data_options(args))
    config_util.setup_render_opts(grid.opt, args)
    with torch.no_grad():
        n_images = dset.n_images
        img_eval_interval = 1
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0
        n_images_gen = 0
        c2ws = dset.c2w.to(device=device)
        lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)

        grid.accelerate()
        grid.opt.use_nearest = False
        grid.opt.step_size *= 2
        grid.opt.stop_thresh = 1e-2  
        all_time = []
        for img_id in tqdm(range(0, n_images, img_eval_interval)):
            dset_h, dset_w = dset.get_image_size(img_id)
            w = dset_w
            h = dset_h

            cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', img_id),
                           dset.intrins.get('fy', img_id),
                           dset.intrins.get('cx', img_id),
                           dset.intrins.get('cy', img_id),
                           w, h,
                           ndc_coeffs=dset.ndc_coeffs)
            


            import time
            torch.cuda.synchronize()
            tic = time.time()
            im = grid.volume_render_image(cam, use_kernel=True, return_raylen=False, nowarp=True)
            im = im.permute(1,0,2)

            # import time
            # torch.cuda.synchronize()
            # tic = time.time()
            # def tree_to_tree_spec(tree, world=True):
            #     """
            #     Pack tree into a TreeSpec (for passing data to C++ extension)
            #     """
            #     from svox2 import csrc as _C
            #     tree_spec = _C.TreeSpec()
            #     tree_spec.data = tree.data
            #     tree_spec.child = tree.child
            #     tree_spec.parent_depth = tree.parent_depth
            #     tree_spec.extra_data = tree.extra_data if tree.extra_data is not None else \
            #             torch.empty((0, 0), dtype=tree.data.dtype, device=tree.data.device)
            #     tree_spec.offset = tree.offset if world else torch.tensor(
            #             [0.0, 0.0, 0.0], dtype=tree.data.dtype, device=tree.data.device)
            #     tree_spec.scaling = tree.invradius if world else torch.tensor(
            #             [1.0, 1.0, 1.0], dtype=tree.data.dtype, device=tree.data.device)
            #     if hasattr(tree, '_weight_accum'):
            #         tree_spec._weight_accum = tree._weight_accum if \
            #                 tree._weight_accum is not None else torch.empty(
            #                         0, dtype=tree.data.dtype, device=tree.data.device)
            #         tree_spec._weight_accum_max = (tree._weight_accum_op == 'max')
            #     return tree_spec
            # tree_spec = tree_to_tree_spec(tree, world=False)

            # im = grid.volume_render_image_tree(cam, tree_spec, use_kernel=True, return_raylen=False, nowarp=True)
            # im = im.permute(1,0,2)




            torch.cuda.synchronize()
            time_delta= time.time()-tic
           
            
            im.clamp_(0.0, 1.0)


            im_gt = dset.gt[img_id].to(device=device)
            mse = (im - im_gt) ** 2
            mse_num : float = mse.mean().item()
            psnr = -10.0 * math.log10(mse_num)
            all_time.append(time_delta)
            print(time_delta, psnr)
            avg_psnr += psnr
            ssim = compute_ssim(im_gt, im).item()
            avg_ssim += ssim
            lpips_i = lpips_vgg(im_gt.permute([2, 0, 1]).contiguous(),
                    im.permute([2, 0, 1]).contiguous(), normalize=True).item()
            avg_lpips += lpips_i
            # from skimage import io
            # io.imsave(f'./imgs/{img_id:03d}.png', im.cpu().numpy())
            im = None
            n_images_gen += 1
        
        avg_psnr /= n_images_gen
        avg_ssim /= n_images_gen
        avg_lpips /= n_images_gen
    print('PSNR\t\tSSIM\t\tLPIPS')
    print(avg_psnr,'\n', avg_ssim,'\n', avg_lpips)
    print(f"average inference time: {sum(all_time)/len(all_time):.4f}")
    average_inference_time = sum(all_time)/len(all_time)
    lines = [f'PSNR\tSSIM\tLPIPS\n']
    lines.append(str(avg_psnr)+'\n'+str(avg_ssim)+'\n'+str(avg_lpips)+'\n' +str(average_inference_time))
    res_savefile = os.path.join(os.path.dirname(ckpt), 'result.txt')
    with open(res_savefile, 'w') as f:
        f.writelines(lines)
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'lpips': avg_lpips
    }

lr_metrics = get_metric(args.ckpt, args.data_dir)