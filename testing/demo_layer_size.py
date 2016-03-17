#!/usr/bin/python
import _caffe_gpu
import numpy as np
import caffe
from net.net_full import net_full as net

im = caffe.io.load_image('../benchmark/MotorRolling/img/0001.jpg')
im = caffe.io.resize(im, (224,224))
im = np.expand_dims(im, 0)
im = im.transpose((0,3,1,2))

im_info = np.array([[im.shape[2], im.shape[3], 1]])

# net.blobs['data'].reshape(*im.shape)
net.forward(data=im, im_info=im_info)

for (name, blob) in net.blobs.iteritems():
    print name, blob.data.shape
