#!/usr/bin/python
import _init_paths
from _net_info import *
import caffe

# full_net

print 'Load Full Net'
net = caffe.Net(ptfile_first_half,model_vgg16,caffe.TEST)


# first_half

# print 'Load First Half'
# net = caffe.Net(ptfile_second_half,model_vgg16,caffe.TEST)


# second half

# print 'Load Second Half'
# net = caffe.Net(ptfile_full,model_vgg16,caffe.TEST)

print 'Load Succeed'
