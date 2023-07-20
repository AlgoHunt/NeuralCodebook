import imp
import os
import os.path as osp
import shutil



def mkdir_p(dir_path):
	if not os.path.isdir(dir_path):
		print('mkdir', dir_path)
		os.mkdir(dir_path)

'''
concat images to video
every time get one image from the render_path
'''
def concatVideo():
	filedir = '/bfs/sz/svox2/opt/ckpt/15fps/'
	tmpdir = '/bfs/sz/svox2/opt/ckpt/tmp/'
	mkdir_p(tmpdir)
	filelist = os.listdir(filedir)
	filelist = [x for x in filelist if x.endswith('.png')]
	ind = 0
	for tim in range(3,50):
		outvd = os.path.join(filedir, f'{int(tim):04d}')
		if not os.path.isdir(outvd):
			print(f'{outvd} does not exist')
			break
		imgpath = os.path.join(outvd, 'test_renders_path1.0x', f'{ind:04}.png')
		tmppath = os.path.join(tmpdir, f'{ind:04}.png')
		if os.path.exists(imgpath):
			print(imgpath)
			shutil.copy(imgpath, tmppath)
			ind += 1
		else:
			print(f'{imgpath} does not exist')
			break
		

if __name__ == '__main__':
	concatVideo()