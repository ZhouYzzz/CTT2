import _init_path
import numpy as np
import caffe
import cv2
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

    # ========================================
    # Environment setting
    # ========================================

    # read vedio sequence
    from utility.read_sequence import read_sequence
    groundtruth, frame_iter, img_shape = \
        read_sequence('benchmark','MotorRolling')

    # init box
    box = groundtruth[0,:]

    # specify layers
    layers = ['pool1', 'conv5_3']

    # init models
    model_xf = None
    model_alphaf = None

    # compute

    # import net
    caffe.set_mode_gpu()
    caffe.set_device(2)

    from testing.net.net_conv import net_conv as net
    print '=========='
    print 'Net Loaded'
    print '=========='

    # ========================================
    # Start tracking
    # ========================================
    from time import time

    RESIZE = False
    FIRST_FRAME = True
    UPDATED = True

    if not RESIZE:
        net.blobs['data'].reshape(*img_shape)

    for frame in frame_iter[:1]:
        print '---- Frame:', frame

        # read frame
        im = cv2.imread(frame)

        tag_img_process = time()

        if RESIZE:
            im = caffe.io.resize(im, (224,224))

        im = np.expand_dims(im, 0)
        im = im.transpose((0,3,1,2))

        print 'time: Image preprocess took', time() - tag_img_process
        
        tag_forward = time()

        # do forward
        blobs_out = net.forward(blobs=layers, data=im)

        print 'time: Net Forward took', time() - tag_forward

        full_feature = blobs_out[layers[0]]

        obj_feature = extract_feature(full_feature, box, img_shape)

        if not FIRST_FRAME:
            FIRST_FRAME = False
            box = predict_location(full_feature, model_xf, model_alphaf)

        if UPDATED:
            pass # do something to adjust the box

        obj_feature = extract_feature(full_feature, box, img_shape)
        model_xf, model_alphaf = \
            update_model(obj_feature, model_xf, model_alphaf)


def predict_location(feat, model_xf, model_alphaf):
    xf = np.fft.fft2(feat)
    kf = np.sum(xf * np.conj(xf), axis=1) / xf.shape[1]
    res_layer = np.real(np.fft.fftshift(np.fft.ifft2(model_alphaf* kzf)))
    response = np.sum(res_layer, axis=1)

    # find traget location
    
    pass

def update_model(feat, yf, model_xf, model_alphaf):
    '''assume feat is (1, 64, 112, 112)'''
    # init
    xf = None
    alphaf = None
    # model update
    # ============
    # fast training
    xf = np.fft.fft2(feat)
    kf = np.sum(xf * np.conj(xf), axis=1) / xf.shape[1]
    alphaf = yf / (kf+ lambda)

    # Model initialization or update
    if not model_xf:
        # first frame, init
        model_alphaf = alphaf
        model_xf = xf
    else:
        interp_factor = 0.01 # learning rate
        model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
        model_xf     = (1 - interp_factor) * model_xf     + interp_factor * xf;

def extract_feature(im_feature, box, img_shape):
    '''return feature around the box'''
    search_box = get_search_box(box, img_shape)
    feat = get_feature(im_feature, search_box)
    return feat

if __name__ == '__main__':
    tracker_ensemble()
