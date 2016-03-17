import _init_paths
# from net.net_second_half import net_second_half as net2
import caffe, cv2
import numpy as np
from time import time

im = caffe.io.load_image('../py-faster-rcnn/data/demo/001150.jpg')
im = caffe.io.resize(im, (224, 224))
im = np.expand_dims(im, 0)
im = im.transpose((0,3,1,2))

caffe.set_mode_gpu()
caffe.set_device(0)

from net.net_first_half  import net_first_half
from net.net_second_half import net_second_half

t = time()
blobs_out = net_first_half.forward(data=im)
print time() - t

conv5_3 = blobs_out['conv5_3']

t = time()
net_second_half.forward(conv5_3=conv5_3, im_info=np.array([[224,224,1]]))
print time() - t
