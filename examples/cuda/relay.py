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

	npoints = 50

	# Generate some random numbers 
	a = np.cos( np.linspace(0,2*m.pi,npoints) )
	a = a.astype(np.float32)

	r = np.array([-0.1,0.4,npoints]).astype(np.float32)

	# Allocate some memory on the device 
	a_gpu = cuda.mem_alloc(a.nbytes)
	r_gpu = cuda.mem_alloc(r.nbytes)

	# Transfer the data to the devcice
	cuda.memcpy_htod(a_gpu, a)
	cuda.memcpy_htod(r_gpu, r)

	# Write some CUDA C code to double each number 
	mod = SourceModule(
		"""
		__global__ void relay(float *a, float *r)
		{
			int idx = threadIdx.x;

			if (idx<r[2]/2){
			
				if (a[idx]<= r[0]+r[1]){
					a[idx] = -1;
				}
			
				else{
					a[idx] = 1;
				}
			}

			else{
			
				if (a[idx]<=r[0]-r[1]){
					a[idx] = -1;
				}
			
				else{
					a[idx] = 1;
				}
			}
		}
		"""
	)	

	# get a reference to the function on device. 
	func = mod.get_function("relay")

	# Call the function on the device on the block we 
	# allocated and transferred "a" to.
	func(a_gpu, r_gpu, block=(npoints,1,1) )

	# Copy the result back into a new array
	a_relay = np.empty_like(a)
	cuda.memcpy_dtoh(a_relay, a_gpu)

	# print results
	plt.plot(a, a_relay)
	plt.show()
