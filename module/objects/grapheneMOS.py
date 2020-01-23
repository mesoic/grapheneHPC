#!/usr/bin/python
#
# Characterization and physical modeling of MOS capacitors in epitaxial 
# graphene monolayers and bilayers on 6H-SiC
#
# AIP Advances 6, 085010 (2016); https://doi.org/10.1063/1.4961361
#
import pycuda.autoinit
import pycuda.driver as cuda

# Utilities
import copy
import random as r

# Numerics and plotting
import numpy as np
import matplotlib.pyplot as plt

# Each CUDA function cals requires the following procedue:
#
# 0) initialize kernel  (cuda.module_from_file) 
# 1) Array creation 	(np.int32, np.float32)
# 2) malloc on GPU  	(cuda.m_alloc)
# 3) copy to GPU 		(cuda.memcpy_htod)
# 4) get GPU function 	(f = mod.get_function)
# 5) call GPU function  (f() )
# 6) copy to CPU 		(cuda.memcpy_dtoh)
# 7) return result 		(return)
#

# GrapheneMOS kernel class (archived) 
class grapheneMOS:

	def __init__(self, NGRID = 16):

		self.mod = cuda.module_from_file("./cuda/graphene.cubin")
			
	#######################################
	#		MONTE CARLO SIMULATION		  #
	#######################################
	def rand_eF(self, _eF, _deF):

		LEN  = np.int32(len(_eF))
		INT  = np.int32(1024)
		SEED = np.int32(r.randint(0, 100000))
		
		eF  = copy.deepcopy(np.array(_eF).astype(np.float32))
		eFr = np.zeros_like(eF).astype(np.float32)
		deF = np.float32(_deF)

		eF_gpu = cuda.mem_alloc(eF.nbytes)
		eFr_gpu = cuda.mem_alloc(eFr.nbytes)

		cuda.memcpy_htod(eF_gpu, eF)
		cuda.memcpy_htod(eFr_gpu, eFr)
		
		RAND_EF = self.mod.get_function("gen_rand")
		RAND_EF(eF_gpu, eFr_gpu, deF, SEED, block=(int(LEN) , 1 , 1 ))

		cuda.memcpy_dtoh(eF, eF_gpu)
		cuda.memcpy_dtoh(eFr, eFr_gpu)
		return eFr

	def CAPACITANCE(self, _eF, _eG, _deF, _Dit, _COX, _M, _DEPTH):
		LEN = np.float32(len(_eF))
		INT  = np.int32(1024)
			   
		eF  = copy.deepcopy(np.array(_eF).astype(np.float32))
		eFr = np.zeros_like(eF).astype(np.float32)
		eG  = copy.deepcopy(np.array(_eG).astype(np.float32))
		deF = copy.deepcopy(np.array(_deF).astype(np.float32))
		Dit = copy.deepcopy(np.array(_Dit).astype(np.float32))
		C   = np.zeros_like(eF).astype(np.float32)
	  
		COX   = np.float32(_COX)
		M	 = np.float32(_M)
		DEPTH = np.int32(_DEPTH)
		SEED  = np.int32(r.randint(0, 100000))

		eF_gpu  = cuda.mem_alloc(eF.nbytes)
		eFr_gpu = cuda.mem_alloc(eFr.nbytes)
		eG_gpu  = cuda.mem_alloc(eG.nbytes)
		deF_gpu  = cuda.mem_alloc(deF.nbytes)
		Dit_gpu = cuda.mem_alloc(Dit.nbytes)
		C_gpu   = cuda.mem_alloc(C.nbytes)

		cuda.memcpy_htod(eF_gpu, eF)
		cuda.memcpy_htod(eFr_gpu, eFr)
		cuda.memcpy_htod(eG_gpu, eG)
		cuda.memcpy_htod(deF_gpu, deF)
		cuda.memcpy_htod(Dit_gpu, Dit)
		cuda.memcpy_htod(C_gpu, C)
			 
		CAPACITANCE = self.mod.get_function("gen_capMC")
		CAPACITANCE(eF_gpu, eFr_gpu, eG_gpu, deF_gpu, Dit_gpu, 
					C_gpu, COX, M, DEPTH, SEED, block=(int(LEN) , 1 , 1 ))

		cuda.memcpy_dtoh(C, C_gpu)
		return C

	#######################################
	#		   NEWTONS METHOD			#
	#######################################
	def SOLVE_EF(self, _v, _vD, _Dit, _eF, _eG, _COX, _TEMP, _M,  FLYWHEEL = 1.0):
	   
		LEN = np.float32(len(_v))
		v   = copy.deepcopy(np.array(_v).astype(np.float32))
		vD  = copy.deepcopy(np.array(_vD).astype(np.float32))
		Dit = copy.deepcopy(np.array(_Dit).astype(np.float32))
		eF  = copy.deepcopy(np.array(_eF).astype(np.float32))
		eG  = copy.deepcopy(np.array(_eG).astype(np.float32))

		v_gpu   = cuda.mem_alloc(v.nbytes)
		vD_gpu  = cuda.mem_alloc(vD.nbytes)
		Dit_gpu = cuda.mem_alloc(Dit.nbytes)
		eF_gpu  = cuda.mem_alloc(eF.nbytes)
		eG_gpu  = cuda.mem_alloc(eG.nbytes)

		M		 = np.float32(_M)
		COX	   = np.float32(_COX)
		TEMP	  = np.float32(_TEMP)
		FLYWHEEL  = np.float32(FLYWHEEL)

		# Host to device
		cuda.memcpy_htod(v_gpu, v)
		cuda.memcpy_htod(vD_gpu, vD)
		cuda.memcpy_htod(Dit_gpu, Dit)
		cuda.memcpy_htod(eF_gpu, eF)
		cuda.memcpy_htod(eG_gpu, eG)
		
		SOLVE_EF = self.mod.get_function("solve_eF")
		SOLVE_EF(v_gpu, vD_gpu, Dit_gpu, eF_gpu, eG_gpu, COX, TEMP, M, FLYWHEEL, block=(int(LEN) , 1 , 1 ) )

		# Device to host
		cuda.memcpy_dtoh(eF, eF_gpu)
		return eF

	#######################################
	#		DENSITY OF STATES			#
	#######################################
	def blDOS(self, _eF, _eG=[-1.0,0.0]): 
		LEN = np.float32(len(_eF))

		eF  = copy.deepcopy(np.array(_eF).astype(np.float32))
		eG  = copy.deepcopy(np.array(_eG).astype(np.float32))
		RO  = np.zeros_like(eF).astype(np.float32)

		eF_gpu  = cuda.mem_alloc(eF.nbytes)
		eG_gpu  = cuda.mem_alloc(eG.nbytes)
		RO_gpu  = cuda.mem_alloc(RO.nbytes)

		cuda.memcpy_htod(eF_gpu, eF)
		cuda.memcpy_htod(eG_gpu, eG)
		cuda.memcpy_htod(RO_gpu, RO)

		CHECK = self.mod.get_function("blDOS_vec")
		CHECK( eF_gpu, eG_gpu, RO_gpu, block=(int(LEN) , 1 , 1 ))
		cuda.memcpy_dtoh(RO, RO_gpu)
		return RO

	def mlDOS(self, _eF): 
		LEN = np.float32(len(_eF))

		eF  = copy.deepcopy(np.array(_eF).astype(np.float32))
		RO  = np.zeros_like(eF).astype(np.float32)

		eF_gpu  = cuda.mem_alloc(eF.nbytes)
		RO_gpu  = cuda.mem_alloc(RO.nbytes)

		cuda.memcpy_htod(eF_gpu, eF)
		cuda.memcpy_htod(RO_gpu, RO)

		CHECK = self.mod.get_function("mlDOS_vec")
		CHECK( eF_gpu, RO_gpu, block=(int(LEN) , 1 , 1 ))
		cuda.memcpy_dtoh(RO, RO_gpu)
		return RO


	#######################################
	#		 CARRIER DENSITIES		   #
	#######################################
	def mlCarriers(self, _eF, _TEMP):
		LEN = np.int32(len(_eF))
		
		eF = copy.deepcopy(np.array(_eF).astype(np.float32))
		n  = np.zeros_like(eF).astype(np.float32)
		p  = np.zeros_like(eF).astype(np.float32)

		eF_gpu = cuda.mem_alloc(eF.nbytes)
		n_gpu  = cuda.mem_alloc(n.nbytes)
		p_gpu  = cuda.mem_alloc(p.nbytes)
		TEMP   = np.float32(_TEMP)

		cuda.memcpy_htod(eF_gpu, eF)
		cuda.memcpy_htod(n_gpu, n)
		cuda.memcpy_htod(p_gpu, p)
			  
		CARRIERS = self.mod.get_function("mlCarriers_vec")
		CARRIERS(eF_gpu, n_gpu, p_gpu, TEMP, block=(int(LEN) , 1 , 1 ))

		cuda.memcpy_dtoh(n, n_gpu)
		cuda.memcpy_dtoh(p, p_gpu)
		return n,p

	def blCarriers(self, _eF, _TEMP, _eG=[-1.0, 0.0]):
		LEN = np.int32(len(_eF))
		
		eF = copy.deepcopy(np.array(_eF).astype(np.float32))
		n  = np.zeros_like(eF).astype(np.float32)
		p  = np.zeros_like(eF).astype(np.float32)
		eG = copy.deepcopy(np.array(_eG).astype(np.float32))
		TEMP   = np.float32(_TEMP)

		eF_gpu = cuda.mem_alloc(eF.nbytes)
		n_gpu  = cuda.mem_alloc(n.nbytes)
		p_gpu  = cuda.mem_alloc(p.nbytes)
		eG_gpu = cuda.mem_alloc(eG.nbytes)

		cuda.memcpy_htod(eF_gpu, eF)
		cuda.memcpy_htod(n_gpu, n)
		cuda.memcpy_htod(p_gpu, p)
		cuda.memcpy_htod(eG_gpu, eG)

		
		CARRIERS = self.mod.get_function("blCarriers_vec")
		CARRIERS(eF_gpu, eG_gpu, n_gpu, p_gpu, TEMP, block=(int(LEN) , 1 , 1 ))

		cuda.memcpy_dtoh(n, n_gpu)
		cuda.memcpy_dtoh(p, p_gpu)
		return n,p

# Monte Carlo Capacitance calculations
if __name__ == "__main__":

		oP = CV_KERNEL()
		_eG = [0.05, 0.02]
		_eF = np.linspace(-0.2, 0.2, 500)

		# Testing Carrier Density Calculation
		if True:
			n1,p1 = oP.mlCarriers(_eF, 300.0) 
			n2,p2 = oP.blCarriers(_eF, 300.0, _eG) 
			n3,p3 = oP.blCarriers(_eF, 300.0) 

			plt.semilogy(_eF,n2)
			plt.semilogy(_eF,p2)
			plt.show()

		# Testing DOS caluclations
		if False:
			mlRHO = oP.mlDOS(_eF) 
			blRHO = oP.blDOS(_eF, _eG) 
			bxRHO = oP.blDOS(_eF) 
			plt.plot(_eF, mlRHO)
			plt.plot(_eF, blRHO)
			plt.plot(_eF, bxRHO)
			plt.show()
