from numpy.fft import *
import numpy as np
from time import time as t
import cv2
layers=['data', 'pool1','pool2','pool3','pool4','conv5_3']

print '=====   RESIZE  ====='
for layer in layers:
    r = np.load('resize/'+layer+'.npy')
    r = r[0,:,:,:]
    print layer, ': SIZE', r.shape, r.shape[0]*r.shape[1]*r.shape[2]

    t_fft = t()
    fft(r)
    print layer, ': FFT took', t()-t_fft

    t_fft2 = t()
    fft2(r)
    print layer, ': FFT2 took', t()-t_fft2

    t_ifft = t()
    ifft(r)
    print layer, ': IFFT took', t()-t_ifft

    t_ifft2 = t()
    ifft2(r)
    print layer, ': IFFT2 took', t()-t_ifft2

    r = r.transpose((1,2,0))
    t_resize = t()
    cv2.resize(r, (224,224))
    print layer, ': resize took', t()-t_resize

print '===== NO_RESIZE ====='
for layer in layers:
    r = np.load('noresize/'+layer+'.npy')
    r = r[0,:,:,:]
    print layer, ': SIZE', r.shape, r.shape[0]*r.shape[1]*r.shape[2]

    t_fft = t()
    fft(r)
    print layer, ': FFT took', t()-t_fft

    t_fft2 = t()
    fft2(r)
    print layer, ': FFT2 took', t()-t_fft2

    t_ifft = t()
    ifft(r)
    print layer, ': IFFT took', t()-t_ifft

    t_ifft2 = t()
    ifft2(r)
    print layer, ': IFFT2 took', t()-t_ifft2

    r = r.transpose((1,2,0))
    t_resize = t()
    cv2.resize(r, (224,224))
    print layer, ': resize took', t()-t_resize
