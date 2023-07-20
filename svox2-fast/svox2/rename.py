import os
import os.path as osp
import shutil

def re1():
	filedir = r'E:\HoloXData\k27v1'
	filelist = []
	for root, dirs, files in os.walk(filedir):
		for name in files:
			filelist.append(os.path.join(root, name))
	os.makedirs(osp.join(filedir, 'depths'), exist_ok=True)
	os.makedirs(osp.join(filedir, 'images'), exist_ok=True)
	for f in filelist:
		suffix = f[len(filedir)+1:]
		suffix = suffix.replace('/', '_')
		suffix = suffix.replace('\\', '_')
		if suffix.endswith('.npy'):
			# print(osp.join(filedir, 'depths', suffix))
			os.rename(f, osp.join(filedir, 'depths', suffix))
		elif suffix.endswith('.png'):
			os.rename(f, osp.join(filedir, 'images', suffix))

# def mkdir_p(d):
#     os.makedirs(d, exist_ok=True)

def mkdir_p(dir_path):
	if not os.path.isdir(dir_path):
		os.mkdir(dir_path)
	return dir_path
'''
rename for E:\HoloXData\15fps\*.png
vnum_camid_rgb.png -> vnum:04d/images/camid:03d_rgb.png
'''
def re2():
	filedir = r'E:\HoloXData\scene5'
	outdir = r'E:\HoloXData\scene5_op'
	filelist = os.listdir(filedir)
	filelist = [x for x in filelist if x.endswith('.png') or x.endswith('.jpg')]
	for f in filelist:
		vnum, camid, suffix = f.split('_')
		# print(vnum, camid, suffix)
		outvd = os.path.join(outdir, f'{int(vnum):04d}')
		mkdir_p(outvd)
		outvd = os.path.join(outvd, 'images')
		mkdir_p(outvd)
		newname = f'{int(camid):03d}_{suffix}'
		newname = os.path.join(outvd, newname)
		oldname = os.path.join(filedir, f)
		os.rename(oldname, newname)
		# print(oldname, newname)

def re3():
	filedir = r'E:\HoloXData\15fps_video'
	xlist = [f'{x:04d}' for x in range(3, 150)]
	for x in xlist:
		d = os.path.join(filedir, x)
		mkdir_p(os.path.join(d,'images'))
		pngs = [x for x in os.listdir(d) if x.endswith('.png')]
		for p in pngs:
			os.rename(os.path.join(d,p), os.path.join(d,'images',p))

def re4():
	filedir = r'E:\HoloXData\scene5_op'
	posefile = r'D:\HoloX\DSVGO\data\video\get_reg\poses_bounds.npy'
	xlist = [f'{x:04d}' for x in range(1, 300)]
	for x in xlist:
		d = os.path.join(filedir, x)
		shutil.copy(posefile, os.path.join(d, 'poses_bounds.npy'))

def re5(filedir, outdir, camnum):
	'''
	filedir: cam_id/time
	outdir: time/cam_id/depths/.npy + 
			time/cam_id/images/.png
	'''
	mkdir_p(outdir)
	# copy images + depths
	# for cam_id in range(camnum):
	# 	d = os.path.join(filedir, str(cam_id)+'_undist')
	# 	files = os.listdir(d)
	# 	for f in files:
	# 		time_id = f.split('_')[0]
	# 		dirpath = mkdir_p(osp.join(outdir, time_id))
	# 		if f.endswith('.png'):
	# 			pngpath = mkdir_p(osp.join(dirpath, 'images'))
	# 			shutil.copy(osp.join(d,f), osp.join(pngpath, f'{cam_id:04d}.png'))
	# 		if f.endswith('.npy'):
	# 			deppath = mkdir_p(osp.join(dirpath, 'depths'))
	# 			shutil.copy(osp.join(d,f), osp.join(deppath, f'{cam_id:04d}.npy'))
	
	# copy pose_bound
	for d in os.listdir(outdir):
		shutil.copy(osp.join(filedir, 'poses_bounds.npy'), osp.join(outdir, d, 'poses_bounds.npy'))
	


if __name__ == '__main__':
	re5(filedir=r'E:\HoloXData\calibrated_data_20220304',
		outdir=r'E:\HoloXData\calibrated_data_20220304\llff',
		camnum=12)