#!/usr/bin/python
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Timing
from datetime import datetime

# Numerics and plotting
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

	# Generate some random numbers 
	a = np.random.randn(4,4)
	a = a.astype(np.float32)

	# Allocate some memory on the device 
	a_gpu = cuda.mem_alloc(a.nbytes)

	# Transfer the data to the devcice
	cuda.memcpy_htod(a_gpu, a)

	# Write some CUDA C code to double each number 
	mod = SourceModule(
		"""
		__global__ void doublify(float *a)
		{
			int idx = threadIdx.x + threadIdx.y*4;
			
			if (a[idx]>0.1)
			{
				 a[idx] *= 2;
			}
		}
		"""
	)

	# get a reference to the function on device. 
	func = mod.get_function("doublify")

	# Call the function on the device on the block we 
	# allocated and transferred "a" to.
	func(a_gpu, block=(4,4,1))

	# Copy the result back into a new array
	a_doubled = np.empty_like(a)
	cuda.memcpy_dtoh(a_doubled, a_gpu)

	# print results
	print(a_doubled)
	print(a)
