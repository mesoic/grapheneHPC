#!/usr/bin/python
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Timing
from datetime import datetime

# Numerics and plotting
import numpy as np
import matplotlib.pyplot as plt

# Hysteresis kernel example
if __name__ == "__main__":

	#################################
	# Set block and grid dimensions #
	#################################
	npoints, ngrid = 1024, 1024 
	bdim	= (npoints, 1, 1)
	gdim	= (ngrid,ngrid)

	# Generate a sin wave 
	a = np.cos( np.linspace(0,2*np.pi,npoints))
	a = a.astype(np.float32)

	# Generate result vector
	r = np.zeros_like(a).astype(np.float32)

	# Generate a meshgrid of relays (energies and widths)
	de	 = 0.25*(max(a)-min(a)) 
	e,d	= np.meshgrid(0.5*np.linspace(min(a), max(a),ngrid),np.linspace(0,de,ngrid))

	e = np.asarray(e).astype(np.float32)
	d = np.asarray(d).astype(np.float32)

	##########################
	#	The Preisach Kernel	 #									
	##########################
	kernel_code_template = """
		__global__ void hysteresis(float* a, float* r, float *e, float* d)
		{
			int idx = %(DIM)s*blockIdx.x + blockIdx.y;
			int tid = threadIdx.x;
			if (tid < %(XLEN)s / 2 )
			{

				if (a[tid] < e[idx] + d[idx])
				{
						r[tid] -= 1.0;
				}
				else{
						r[tid] += 1.0;
				}
			}
			
			else{
			
				if (a[tid] < e[idx] - d[idx])
				{
						r[tid] -= 1.0;
				}
				else
				{
						r[tid] += 1.0;
						//r[i+ %(XLEN)s*idx] += 1.0;
				}
			}
			__syncthreads();
		}
	"""

	# Preprocessor
	kernel_code = kernel_code_template % {
		'XLEN': npoints,
		'DIM' : ngrid,
		'd'	  : '%d'
	}
	mod = SourceModule(kernel_code)
	func = mod.get_function("hysteresis")

	tic = datetime.now()
	############################################
	a_gpu = cuda.mem_alloc(a.nbytes) 
	r_gpu = cuda.mem_alloc(r.nbytes) 
	e_gpu = cuda.mem_alloc(e.nbytes)
	d_gpu = cuda.mem_alloc(d.nbytes)

	cuda.memcpy_htod(a_gpu, a)
	cuda.memcpy_htod(r_gpu, r)
	cuda.memcpy_htod(e_gpu, e)
	cuda.memcpy_htod(d_gpu, d)

	func(a_gpu, r_gpu,	e_gpu, d_gpu, block=bdim, grid=gdim)

	cuda.memcpy_dtoh(r, r_gpu)
	############################################
	toc = datetime.now()
	d	= toc-tic

	print("%d Hysterions evaluated on %d points in %dus"%(ngrid*ngrid,npoints,d.microseconds))

	# print results
	plt.plot(a,r)
	plt.show()
