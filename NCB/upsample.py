import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from svox2 import utils
_C = utils._get_c_extension()


def tile_process(grid_4xsh,
                 grid_2xd,
                 model,
                 device,
                 scale=4,
                 tile_size=64,
                 tile_pad=10):
    grid_4xsh, grid_2xd = grid_4xsh, grid_2xd
    batch, channel, depth, height, width = grid_4xsh.shape
    output_height = height * scale
    output_width = width * scale
    output_depth = depth * scale
    output_shape = (batch, output_height, output_width, output_depth, channel)
    output_shape_den = (batch, output_height, output_width, output_depth, 1)

    # start with black image
    output_sh = grid_4xsh.new_zeros(output_shape).half().to(device)
    output_den = grid_2xd.new_zeros(output_shape_den).to(device)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    tiles_z = math.ceil(depth / tile_size)

    # loop over all tiles
    for z in range(tiles_z):
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                ofs_z = z * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)
                input_start_z = ofs_z
                input_end_z = min(ofs_z + tile_size, depth)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)
                input_start_z_pad = max(input_start_z - tile_pad, 0)
                input_end_z_pad = min(input_end_z + tile_pad, depth)

                input_start_x_pad_2x = input_start_x_pad * 2
                input_end_x_pad_2x = input_end_x_pad * 2
                input_start_y_pad_2x = input_start_y_pad * 2
                input_end_y_pad_2x = input_end_y_pad * 2
                input_start_z_pad_2x = input_start_z_pad * 2
                input_end_z_pad_2x = input_end_z_pad * 2

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                input_tile_depth = input_end_z - input_start_z

                tile_idx = y * tiles_x + x + 1
                cropped_sh = grid_4xsh[:, :, input_start_z_pad:input_end_z_pad, input_start_y_pad:input_end_y_pad,
                                     input_start_x_pad:input_end_x_pad]

                cropped_2xden = grid_2xd[:, :, input_start_z_pad_2x:input_end_z_pad_2x,
                                            input_start_y_pad_2x:input_end_y_pad_2x,
                                            input_start_x_pad_2x:input_end_x_pad_2x]

                input_tile = [cropped_2xden.to(device), cropped_sh.to(device)]
                try:
                    with torch.no_grad():
                        output_tile = model(input_tile)
                except Exception as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale
                output_start_z = input_start_z * scale
                output_end_z = input_end_z * scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale
                output_start_z_tile = (input_start_z - input_start_z_pad) * scale
                output_end_z_tile = output_start_z_tile + input_tile_depth * scale

                z_max = torch.max(output_tile[1][0, :, :, :])
                z_min = torch.min(output_tile[1][0, :, :, :])
                print('z_max is :{}, z_min is :{}'.format(z_max, z_min))

                # put tile into output image
                if output_tile[0] is not None:
                    output_sh[:, output_start_y:output_end_y, output_start_x:output_end_x,
                              output_start_z:output_end_z, :] = output_tile[
                                  0][:, :, output_start_z_tile:output_end_z_tile, output_start_y_tile:output_end_y_tile,
                                     output_start_x_tile:output_end_x_tile].half().permute(0, 3, 4, 2, 1).contiguous()
                if output_tile[1] is not None:
                    output_den[:, output_start_y:output_end_y, output_start_x:output_end_x,
                               output_start_z:output_end_z, :] = output_tile[1][:, :,
                                                                                output_start_z_tile:output_end_z_tile,
                                                                                output_start_y_tile:output_end_y_tile,
                                                                                output_start_x_tile:output_end_x_tile].permute(0, 3, 4, 2, 1).contiguous()
    return [output_sh, output_den]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, choices=['SYN', 'LLFF'], default='SYN', help='The categroy of test dataset.')
    parser.add_argument('--data_name', type=str, required=True, help='dataset name.')
    parser.add_argument('--model_name', type=str, required=True, help='name of the upsampling module.')
    parser.add_argument('--up_mode', required=True, choices=['Tri', 'SR'], default='SYN', help='upsampling mode.')
    parser.add_argument('--tile_size_lr', type=int, default=16, help='name of the upsampling module.')
    args = parser.parse_args()

    data_root = args.data_root
    data_name = args.data_name
    model_name = args.model_name
    model_path = 'pretrained/{}'.format(model_name)
    device = 'cuda:0'
    tile_size_lr = args.tile_size_lr
    input_4xsh = '../data_voxel/{0}/{1}/ckpt/ckpt_{1}_4xsh.npz'.format(data_root, data_name)
    input_4xindex = '../data_voxel/{0}/{1}/ckpt/ckpt_{1}_4x_index.npz'.format(data_root, data_name)
    input_2xd = '../data_voxel/{0}/{1}/ckpt/ckpt_{1}_2xd.npz'.format(data_root, data_name)
    input_2xindex = '../data_voxel/{0}/{1}/ckpt/ckpt_{1}_2x_index.npz'.format(data_root, data_name)
    # #####################################################################
    input_hrindex = '../data_voxel/{0}/{1}/ckpt/ckpt_{1}_dense_bitlink.npz'.format(data_root, data_name)
    output = '../data_voxel/{0}/{1}/ckpt/'.format(data_root, data_name)

    os.system('unzip {0}'.format(input_4xindex.replace('.npz', '.zip')))
    os.system('unzip {0}'.format(input_2xindex.replace('.npz', '.zip')))
    os.system('unzip {0}'.format(input_hrindex.replace('.npz', '.zip')))
    scale_up = 4

    if data_root == 'SYN':
        voxel_size = np.array([512, 512, 512])
        center = np.array([0., 0., 0.])
        radius = np.array([1., 1., 1.])
        basis_type = np.array(1)

    elif data_root == 'LLFF':
        voxel_size = np.array([1408, 1156, 128])
        center = np.array([0., 0., 0.])
        radius = np.array([1.4960318, 1.6613756, 1.0])
        basis_type = np.array(1)

    os.makedirs(output, exist_ok=True)

    imgname, extension = os.path.splitext(os.path.basename(input_4xsh))
    print('Testing', imgname)

    grid_4x = np.load(input_4xsh)
    grid_2x = np.load(input_2xd)

    index_hr = np.load(input_hrindex)
    index_hr = index_hr.f.arr_0
    index_hr = np.reshape(np.unpackbits(index_hr), voxel_size).astype(np.bool)
    index_4x = np.load(input_4xindex)
    index_4x = index_4x.f.arr_0
    index_4x = np.reshape(np.unpackbits(index_4x), voxel_size // 4)
    index_2x = np.load(input_2xindex)
    index_2x = index_2x.f.arr_0
    index_2x = np.reshape(np.unpackbits(index_2x), voxel_size // 2)

    index_4x = np.reshape(index_4x, -1) == 1
    full_4xsh = np.zeros((index_4x.shape[0], 27)).astype(np.float)
    full_4xsh[index_4x] = grid_4x.f.arr_0
    full_4xsh = np.reshape(full_4xsh, (voxel_size[0] // 4, voxel_size[1] // 4, voxel_size[2] // 4, 27))

    index_2x = np.reshape(index_2x, -1) == 1
    full_2xd = np.zeros((index_2x.shape[0], 1)).astype(np.float)
    full_2xd[index_2x] = grid_2x.f.arr_0
    full_2xd = np.reshape(full_2xd, voxel_size // 2)

    print('density', full_2xd.shape, full_2xd.dtype)
    print('sh', full_4xsh.shape, full_4xsh.dtype)

    grid_2xden = full_2xd
    grid_2xd = torch.from_numpy(grid_2xden.transpose(2, 0, 1)).unsqueeze(0).unsqueeze(0).float()
    grid_4xsh = full_4xsh
    grid_4xsh = torch.from_numpy(grid_4xsh.transpose(3, 2, 0, 1)).unsqueeze(0).float()

    try:
        if args.up_mode == 'SR':
            model = vx_R3CAN.VX_R3CAN(
                    n_colors_i=27,
                    n_colors_o=27,
                    n_resgroups=4,
                    n_resblocks=6,
                    n_feats=81,
                    reduction=9,
                    finetine=False,
                    cp_mode='mul',
                    scale=scale_up,
                    act_type='Leaky')

            print('Loading ckpt from {}...'.format(model_path))
            loadnet = torch.load(model_path)
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            else:
                keyname = 'params'
            model.load_state_dict(loadnet[keyname], strict=True)
            model.eval()
            # name_model = os.path.splitext(os.path.basename(model_path))[0]

            model = model.to(device)
            # if args.half:
            #     model = model.half()
            [output_sh, pred_den] = tile_process(grid_4xsh,
                                                grid_2xd,
                                                model,
                                                device,
                                                tile_size=tile_size_lr,
                                                scale=scale_up)
            torch.cuda.empty_cache()
        elif args.up_mode == 'Tri':
            output_sh = F.interpolate(grid_4xsh, scale_factor=4, mode='trilinear').permute(0, 3, 4, 2, 1).contiguous()
            pred_den = F.interpolate(grid_2xd, scale_factor=2, mode='trilinear').permute(0, 3, 4, 2, 1).contiguous()
    except Exception as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    else:
        extension = extension[1:]
        save_path = os.path.join(output, f'{imgname}_{args.up_mode}.{extension}')
        output_sh = output_sh.squeeze()

        output_den = pred_den.squeeze(0)

        torch.cuda.empty_cache()
        index_hr = np.reshape(index_hr, -1)
        output_den = output_den.view(-1, 1).to('cpu')
        output_den = output_den[index_hr]
        output_sh = output_sh.view(-1, 27).to('cpu').half()
        output_sh = output_sh[index_hr, :]

        init_links = (
            torch.cumsum(torch.from_numpy(index_hr).to(torch.int32), dim=-1).int() - 1)
        init_links[~index_hr] = -1
        init_links = init_links.view(voxel_size.tolist()).to(device=device)

        assert (_C is not None and init_links.is_cuda
                ), "CUDA extension is currently required for accelerate"
        _C.accel_dist_prop(init_links)

        output_dic = {
            "center": center,
            "radius": radius,
            "basis_type": basis_type,
            "links": init_links.cpu().numpy(),
            "sh_data": output_sh,
            "density_data": output_den}
        np.savez(save_path, **output_dic)


if __name__ == '__main__':
    main()
