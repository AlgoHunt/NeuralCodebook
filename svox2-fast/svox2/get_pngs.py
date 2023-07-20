import os
import os.path as osp
from shutil import copyfile

def solve():
    pass

if __name__ == "__main__":
    root_dir = r'D:\HoloX\svox2\opt\ckpt\scene3_colevery'
    out_dir = r'D:\HoloX\svox2\opt\sr\sr_person'
    filelist = []
    for root, dirs, files in os.walk(root_dir):
        # if 'train_renders_0.5x' in root:
        for name in files:
            if name.endswith('.png'):
                filelist.append(os.path.join(root, name))
    print(len(filelist))
    for f in filelist:
        if 'train_renders0.5x' in f:
            res = '360p'
            continue
        elif 'train_renders1.0x' in f:
            res = '720p'
            continue
        elif 'test_renders_path0.5x' in f:
            res = '360p_test'
        else:
            continue
        suffix = f[len(root_dir)+1:]
        suffix = suffix.replace('/', '_')
        suffix = suffix.replace('\\', '_')
        filename = osp.join(out_dir, res, suffix)
        # print( f, filename)
        copyfile(f, filename)
        assert os.path.exists(f)
        # break