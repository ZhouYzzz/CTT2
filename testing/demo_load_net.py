#!/usr/bin/python
import _init_paths
from _net_info import *
import caffe

# full_net

print 'Load Full Net'
net = caffe.Net(full_net_ptfile,net_model,caffe.TEST)


# first_half

# print 'Load First Half'
# net = caffe.Net(first_half_ptfile,net_model,caffe.TEST)


# second half

# print 'Load Second Half'
# net = caffe.Net(second_half_ptfile,net_model,caffe.TEST)

print 'Load Succeed'
