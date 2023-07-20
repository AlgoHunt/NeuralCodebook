import svox2
import torch
grid = svox2.SparseGrid()

# point only visible to x
for i in range(20):
	print((grid.viewcount_helper == 2**i).count_nonzero(-1).sum(-1))


# point been visible to x
for i in range(20):
    visible_to_cuerrent_view = (grid.viewcount_helper & 2**i)!=0
    print((visible_to_cuerrent_view * view_count**2).float().sum()/visible_to_cuerrent_view.sum() )

# point been visible to i, and have been seen by 2
for vc_thres in range(19):
    print(f"vc_thres {vc_thres}")
    for i in range(19):
        visible_to_cuerrent_view = (grid.viewcount_helper & 2**i)!=0
        print((visible_to_cuerrent_view & (view_count>=vc_thres)).sum()/visible_to_cuerrent_view.sum())


for i in range(19):
        visible_to_cuerrent_view = (grid.viewcount_helper & 2**i)!=0
        print((visible_to_cuerrent_view & (view_count>=)).sum()/visible_to_cuerrent_view.sum())


for i in range(20):
    print((((grid.viewcount_helper & 2**i) !=0)).sum())

rgb_diff = rgb_cache - c_rgb
rgb_diff = abs(rgb_diff).sum(-1)/3
rgb_diff = (rgb_diff>0.005)
print(rgb_diff.sum()/rgb_diff.shape[0])

import tensorflow as tf
loss = tf.reduce_mean(tf.square(1))
optimizer = tf.tra

