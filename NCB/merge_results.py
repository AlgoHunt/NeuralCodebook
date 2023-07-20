import os
import argparse
from prettytable import PrettyTable

par = argparse.ArgumentParser()
par.add_argument('resdir', type=str)

args = par.parse_args()
resdir = args.resdir

exps = os.listdir(resdir)

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

table = PrettyTable(["Scene", "PSNR", "SSIM", "LPIPS"])

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
    PSNR.update(psnr)
    SSIM.update(ssim)
    LPIPS.update(lpips)
    table.add_row([exp, psnr, ssim, lpips])
table.add_row(['Mean', PSNR.avg, SSIM.avg, LPIPS.avg])
writedir = os.path.join(resdir, 'merge.txt')
with open(writedir, 'w') as f:
    f.writelines(table.get_string())

print(table)