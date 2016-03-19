import _init_path
import numpy as np
# import caffe

''' VGG16:

data    (1, 3, 224, 224)
-------------------------
conv1_1 (1, 64, 224, 224)
conv1_2 (1, 64, 224, 224)
pool1   (1, 64, 112, 112)
-------------------------
conv2_1 (1, 128, 112, 112)
conv2_2 (1, 128, 112, 112)
pool2   (1, 128, 56, 56)
-------------------------
conv3_1 (1, 256, 56, 56)
conv3_2 (1, 256, 56, 56)
conv3_3 (1, 256, 56, 56)
pool3   (1, 256, 28, 28)
-------------------------
conv4_1 (1, 512, 28, 28)
conv4_2 (1, 512, 28, 28)
conv4_3 (1, 512, 28, 28)
pool4   (1, 512, 14, 14)
-------------------------
conv5_1 (1, 512, 14, 14)
conv5_2 (1, 512, 14, 14)
conv5_3 (1, 512, 14, 14)

'''

def tmp_imread(img):
    return np.zeros([224,224,3])

def tracker_ensemble():
    '''Run Tracker'''
    global net
    # ========================================
    # Environment setting
    # ========================================

    # read vedio sequence
    from utility.read_sequence import read_sequence
    groundtruth, frame_iter = read_sequence('benchmark','MotorRolling')

    layers = ['conv2_1', 'conv5_3']

    # ========================================
    # Start tracking
    # ========================================
    for frame in frame_iter:
        # read frame
        # im = caffe.io.imread(frame)
        im = tmp_imread(frame)

        im = reshape(im)

        # do forward
        blobs_out = net.forward(blobs=layers, data=im)



if __name__ == '__main__':
    tracker_ensemble()