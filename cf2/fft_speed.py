from numpy.fft import *
import numpy as np
from time import time as t

a = np.load('resize/pool3.npy')
t1 = t()
a = a[0,:,:,:]
a = a.astype(np.complex64)
print t()-t1
print a.shape


t1 = t()
ifft2(a)
print 'NUMPY took', t()-t1

from pyfft.cuda import Plan
import pycuda.driver as cuda
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray

cuda.init()
context = make_default_context()
stream = cuda.Stream()

plan = Plan((128,128), stream=stream)

t1 = t()
gpu_data = gpuarray.to_gpu(a)
print 'togpu took', t()-t1
plan.execute(gpu_data)
result = gpu_data.get()

gpu_data = gpuarray.to_gpu(a)
plan.execute(gpu_data)
result = gpu_data.get()

# np.conj(result)
print 'CUDA took', (t()-t1)/2, result.shape

t2 = t()
result*np.conj(result)
print 'AFTER took', t()-t2

t2 = t()
g = gpuarray.to_gpu(result)

print 'AFTER took', t()-t2

context.pop()
