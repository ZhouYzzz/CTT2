import sys
sys.path.insert(0,'..')

import _init_paths
from _net_info import *
import caffe

net_conv = caffe.Net(ptfile_conv, model_vgg16,caffe.TEST)
