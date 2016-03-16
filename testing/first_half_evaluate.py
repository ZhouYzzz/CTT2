import _init_paths
from _net_info import *
import caffe, cv2
from fast_rcnn.config import cfg
from fast_rcnn.test import _get_blobs
import numpy as np
from time import time

im = caffe.io.load_image('../py-faster-rcnn/data/demo/001150.jpg')
im = caffe.io.resize(im, (224, 224))
im = np.expand_dims(im, 0)
im = im.transpose((0,3,1,2))

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(ptfile_first_half,model_vgg16,caffe.TEST)

t = time()
blobs_out = net.forward(data=im, im_info=np.array([[224,224,1]]))
np.save('conv5_3', blobs_out['conv5_3'])
print time() - t