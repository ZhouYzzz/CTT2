import os.path as osp

_tfp = osp.dirname(__file__)

# faster_rcnn_test.pt  first_half.pt  second_half.pt

# ../py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel

model_vgg16 = osp.join(_tfp, '..', 'py-faster-rcnn', 'data', 'faster_rcnn_models', 'VGG16_faster_rcnn_final.caffemodel')

ptfile_first_half = osp.join(_tfp, 'ptfile', 'first_half.pt')
ptfile_second_half = osp.join(_tfp, 'ptfile', 'second_half.pt')
ptfile_full = osp.join(_tfp, 'ptfile', 'faster_rcnn_test.pt')
