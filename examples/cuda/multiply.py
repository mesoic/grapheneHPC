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

	mod = SourceModule(
		"""
		__global__ void multiply(float *dest, float *a, float *b)
		{
			const int i = threadIdx.x;
			dest[i] = a[i] * b[i];
		}
		"""
	)

	multiply = mod.get_function("multiply")

	a = numpy.random.randn(400).astype(numpy.float32)
	b = numpy.random.randn(400).astype(numpy.float32)

	result = numpy.zeros_like(a)
	multiply(
		cuda.Out(result), 
		cuda.In(a), 
		cuda.In(b), 
		block=(400,1,1), 
		grid=(1,1)
	)

	print(result, a, b)
