import numpy as np
# from scipy.misc import imresize

a = np.load('conv1_2.npy')
# a = np.load('conv5_3.npy')
print  a.shape

from time import time

t = time()
i = a[0,:,:,:].transpose((1,2,0))
print time()-t

print i.shape

# from skimage.transform import pyramid_expand
# from scipy.ndimage import map_coordinates
import cv2

t = time()
# r = imresize(i, (28,28), mode='F')
# for x in xrange(10):
r = cv2.resize(i,(448,448))

sum = np.sum(r,axis=2)

print sum.shape

print r.shape, time()-t
