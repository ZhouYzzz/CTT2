import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

THIS_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(THIS_DIR, '..')
CAFFE_DIR = osp.join(ROOT_DIR, 'py-faster-rcnn', 'caffe-fast-rcnn', 'python')
FRCNN_DIR = osp.join(ROOT_DIR, 'py-faster-rcnn', 'lib')

add_path(ROOT_DIR)
add_path(CAFFE_DIR)
add_path(FRCNN_DIR)