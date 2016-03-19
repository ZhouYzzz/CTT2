CAFFE_PATH = '~/caffe/matlab/';
addpath(CAFFE_PATH);

CTT2_ROOT = '~/proj/CTT2/';

% model_vgg16 = osp.join(_tfp, '..', 'py-faster-rcnn', 'data', 'faster_rcnn_models', 'VGG16y')

ptfile = [CTT2_ROOT 'testing/ptfile/conv.pt'];
model = [CTT2_ROOT 'py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel'];


caffe.set_mode_gpu();
caffe.set_device(0);

net = caffe.Net(ptfile, model, 'test');

net
