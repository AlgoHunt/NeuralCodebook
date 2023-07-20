import sys
import os
import argparse
from multiprocessing import Process, Queue
from typing import List, Dict
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", "-g", type=str, required=True,
                            help="space delimited GPU id list (global id in nvidia-smi, "
                                 "not considering CUDA_VISIBLE_DEVICES)")
parser.add_argument('--eval', action='store_true', default=False,
                   help='evaluation mode (run the render_imgs script)')
parser.add_argument('--dataset', type=str, choices=['syn', 'llff'])
args = parser.parse_args()

PSNR_FILE_NAME = 'test_psnr.txt'

def run_exp(env, input_ckpt, target_ckpt, data_dir, save_dir, config, importance_sample=False):
    base_cmd = ['python', 'train.py', input_ckpt, target_ckpt, data_dir, '-t', save_dir,
            '-c', config]
    if importance_sample:
        base_cmd.append(" --use_importance_threshold --threshold_percent 0.05")
    psnr_file_path = os.path.join(save_dir, PSNR_FILE_NAME)
    if os.path.isfile(psnr_file_path):
        print('! SKIP', save_dir, "on ", env["CUDA_VISIBLE_DEVICES"])
        return
    log_file_path = os.path.join(save_dir, 'log')
    print('********************************************')
    opt_cmd = ' '.join(base_cmd)
    print(opt_cmd, "on ", env["CUDA_VISIBLE_DEVICES"])
    try:
        opt_ret = subprocess.check_output(opt_cmd, shell=True, env=env).decode(
                sys.stdout.encoding)
    except subprocess.CalledProcessError:
        print('Error occurred while running TRAIN for exp', save_dir, 'on', env["CUDA_VISIBLE_DEVICES"])
        return
    finally:
        with open(log_file_path, 'a+') as f:
            f.write(opt_ret)

def run_eval(env, input_ckpt, target_ckpt, data_dir, save_dir, config, importance_sample=False):
    base_cmd = ['python', 'eval.py', os.path.join(save_dir,'tune.npz'), data_dir,
            '-c', config]
   
    psnr_file_path = os.path.join(save_dir, PSNR_FILE_NAME)
    if os.path.isfile(psnr_file_path):
        print('! SKIP', save_dir, "on ", env["CUDA_VISIBLE_DEVICES"])
        return
    log_file_path = os.path.join(save_dir, 'log')
    print('********************************************')
    opt_cmd = ' '.join(base_cmd)
    print(opt_cmd, "on ", env["CUDA_VISIBLE_DEVICES"])
    try:
        opt_ret = subprocess.check_output(opt_cmd, shell=True, env=env).decode(
                sys.stdout.encoding)
    except subprocess.CalledProcessError:
        print('Error occurred while running TRAIN for exp', save_dir, 'on', env["CUDA_VISIBLE_DEVICES"])
        return
    finally:
        with open(log_file_path, 'a+') as f:
            f.write(opt_ret)


def process_main(device, queue):
    # Set CUDA_VISIBLE_DEVICES programmatically
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    while True:
        task = queue.get()
        if len(task) == 0:
            break
        if args.eval:
            run_eval(env, **task)
        else:
            run_exp(env, **task)

pqueue = Queue()

DatasetSetting={
    "syn": {
        "model": "/bfs/AAAIResearch/NCB_res_nerfsyn",
        "data": "/bfs/HoloResearch/NeRFData/nerf_synthetic",
        "cfg": "configs/syn.json",
        "save_dir": "./NCB_result/ablation/weight_0"
    },
    "llff":{
        "model": "./assets/NCB_res_llff",
        "data": "./dataset/nerf_llff_data",
        "cfg": "configs/llff.json",
        "save_dir": "./NCB_result/llff"
    }
}

datasetting = DatasetSetting[args.dataset]
all_tasks = []
for setting in ['with_imp']:
    for scene in ['lego', 'chair', 'ficus', 'hotdog', 'mic', 'ship',  'materials', 'drums']:
        task: Dict = {}
        task['input_ckpt'] = f'{datasetting["model"]}/{scene}/{scene}_TRI.npz'
        task['target_ckpt'] = f'{datasetting["model"]}/{scene}/{scene}_hr.npz'
        task['data_dir'] = f'{datasetting["data"]}/{scene}'
        task['save_dir'] = f'{datasetting["save_dir"]}/{scene}_{setting}'
        if setting == 'with_imp':
            task["importance_sample"] = True
        task["config"] = datasetting['cfg']
        assert os.path.exists(task['data_dir']), task['data_dir'] + ' does not exist'
        assert os.path.isfile(task['input_ckpt']), task['input_ckpt'] + ' does not exist'
        assert os.path.isfile(task['target_ckpt']), task['target_ckpt'] + ' does not exist'
        assert os.path.isfile(task['config']), task['config'] + ' does not exist'
        all_tasks.append(task)

for task in all_tasks:
    pqueue.put(task)

args.gpus = list(map(int, args.gpus.split()))
print('GPUS:', args.gpus)

for _ in args.gpus:
    pqueue.put({})

all_procs = []
for i, gpu in enumerate(args.gpus):
    process = Process(target=process_main, args=(gpu, pqueue))
    process.daemon = True
    process.start()
    all_procs.append(process)

for i, gpu in enumerate(args.gpus):
    all_procs[i].join()



resdir = datasetting['save_dir']
exps = os.listdir(resdir)
writelines = []

class AverageMeter(object):
    def __init__(self, name=''):
        self.name=name
        self.reset()
    def reset(self):
        self.val=0
        self.sum=0
        self.avg=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum += val*n
        self.count += n
        self.avg=self.sum/self.count
    def __repr__(self) -> str:
        return f'{self.name}: average {self.count}: {self.avg}\n'

PSNR=AverageMeter('PSNR')
SSIM=AverageMeter('SSIM')
LPIPS=AverageMeter('LPIPS')
TESTTIME=AverageMeter('TESTTIME')
from prettytable import PrettyTable
table = PrettyTable(["Scene", "PSNR", "SSIM", "LPIPS","TESTTIME"])

for exp in exps:
    res = os.path.join(resdir, exp, 'result.txt')
    if not os.path.exists(res):
        continue
    with open(res, 'r') as f:
        lines = f.readlines()
    # parse the metrics
    psnr = float(lines[1].strip())
    ssim = float(lines[2].strip())
    lpips = float(lines[3].strip())
    inference_time = float(lines[4].strip())
    PSNR.update(psnr)
    SSIM.update(ssim)
    LPIPS.update(lpips)
    TESTTIME.update(inference_time)
    table.add_row([exp, psnr, ssim, lpips, inference_time])
table.add_row(['Mean', PSNR.avg, SSIM.avg, LPIPS.avg, TESTTIME.avg])
writedir = os.path.join(resdir, 'merge.txt')
with open(writedir, 'w') as f:
    f.writelines(table.get_string())

print(table)