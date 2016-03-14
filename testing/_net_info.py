import os.path as osp

_tfp = osp.dirname(__name__)

# faster_rcnn_test.pt  first_half.pt  second_half.pt

# ../py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel

net_model = osp.join(_tfp, '..', 'py-faster-rcnn', 'data', 'faster_rcnn_models', 'VGG16_faster_rcnn_final.caffemodel')

first_half_ptfile = osp.join(_tfp, 'ptfile', 'first_half.pt')
second_half_ptfile = osp.join(_tfp, 'ptfile', 'first_half.pt')
full_net_ptfile = osp.join(_tfp, 'ptfile', 'faster_rcnn_test.pt')
