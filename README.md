# Compact Real-Time Radiance Fields With Neural Codebook

## Overview
This repository contain three parts:

1) *svox2*: modified version of Plenoxels <https://github.com/sxyu/svox2>. 

3) *NCB*： training and inference code for neural codebook.


## Setup
We recommend using conda to setup environment:
```sh
conda env create -f environment.yml
conda activate plenoxel
```


If your CUDA toolkit is older than 11, then you will need to install CUB as follows:
`conda install -c bottler nvidiacub`.
Since CUDA 11, CUB is shipped with the toolkit.

To install the modified svox2, simply run
```
cd svox2-fast
pip install .
```


## Getting datasets

Please get two datasets used in our experiment from svox2's mainpage:

<https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1 (nerf_synthetic.zip and nerf_llff_data.zip)>

After download these datasets, please put them into the `data_voxel/` dir, and makesure NeRF-synthetic in the `data_voxel/SYN/` dir, and LLFF datasets in the `data_voxel/LLFF/` dir.

## Getting pretrained plenoxels

Download pretrained plenoxels in:
 https://drive.google.com/drive/folders/1SOEJDw8mot7kf5viUK9XryOAmZGe_vvE?usp=sharing

## Non-uniform compression
1. Swith io NCB subdirectory:

    ```cd NCB```.

2. Convert original sparse voxels grid to a dense voxels grid 

    `python process_vox.py --data_root SYN --data_name lego --mode recover_dense`

3. Downsample and convert it back to sparse voxels grid:

    `python process_vox.py --data_root SYN --data_name lego --mode spread_lr`

After compression we will get ： 
1. original volume grid 
2. 2x downsampled sparse density data 
3. 4x downsampled sparse spherical harmonic coefficients data 
4. mask for (2)   
5. mask for (3) 

in directory `data_voxel/ckpt/`


## Nerual Codebook Training

1. Swith io NCB subdirectory:

    `cd NCB`

2. Pre-calculate importance for reweighting

    `python calc_importance.py -t ckpt/lego ./nerf_systhesis_data/lego -c configs/lego.json --pretrained <hr_ckpt>`

an `importance_map.pth` will be saved in the same directroy with pretrained model.

3. Train the neural codebook

    `python train.py <sr_ckpt> <hr_ckpt> <data_dir> -t <save_dir> --use_lowres --lr_ckpt <lr_ckpt> --use_importance_map`

the SH and density codebook weights  will be saved in `save_dir/CodebookNet.pth`
the compressed model will be saved in `save_dir/tune.npz`


## Decompression

1. Swith io NCB subdirectory:

    `cd NCB`

2. Upsample the downsampled voxels grid:

    `python upsample.py --data_root SYN --data_name lego --up_mode Tri`

3. Refine & evaluate with neural codebook

    `python eval.py <path/to/compressedmodel> <data_dir> -c <config_file>`

all result include `PSNE/SSIM/LPIPS` will write into `results.txt` in the same directory with the compressed model.



# Batch Training & Testing

Set the dataset setting in the **autotask.py**


launch nerf-synthetic experiments
```python
python autotask -g "0 1 2 3 4 5 6 7" -llff
```

launch LLFF experiments
```python
python autotask -g "0 1 2 3 4 5 6 7" -llff
```

launch nerf-synthetic evaluate
```python
python autotask -g "0 1 2 3 4 5 6 7" --syn --eval
```

launch LLFF evaluate
```python
python autotask -g "0 1 2 3 4 5 6 7" --llff --eval
```


<!-- ## Evaluation

Use the `opt/render_imgs.py` in svox2, see the log file for evaluation result. -->
