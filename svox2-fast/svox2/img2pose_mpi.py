import os
from concurrent.futures import ThreadPoolExecutor
def action(x):
	cmd = f'python imgs2poses.py ../svox2/opt/data/video/15fps_video/{x:04d}'
	os.system(cmd)
	return cmd
	
with ThreadPoolExecutor(max_workers=16) as pool:
	tasks = (x for x in range(3,150))
	results = pool.map(action, tasks)
	print('--------------')
	for r in results:
		print(r)
