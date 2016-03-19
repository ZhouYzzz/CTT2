import os.path as osp
_tfp = osp.dirname(__file__)
import sys
sys.path.insert(0,_tfp+'/..')

import _init_paths
from _net_info import *
import caffe

net_full = caffe.Net(ptfile_full,model_vgg16,caffe.TEST)
