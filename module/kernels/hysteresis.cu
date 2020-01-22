//////////////////////////////////////////////////////////////////////////
//
//	This program is free software: you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation, either version 3 of the License, or
//	(at your option) any later version.
//
//	This program is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
//	GNU General Public License for more details.
//
//	You should have received a copy of the GNU General Public License
//	along with this program.	If not, see <http://www.gnu.org/licenses/>.
//
// Author: M. Winters (2016)
// Institution: Chalmers Universtiy of Technology (MC2)
// email: mesoic@protonmail.com 
//
//////////////////////////////////////////////////////////////////////////
//
// Hysteresis Modeling Kernel
//
// Compile with: nvcc -c -arch=sm_20 <filename>.cu	
// 
// Compilation will create CV_kernel.cubin which provides the functions below. 
// The .cubin can then be imported into Python(PyCUDA) or C/C++ code.	
// 
// This kernel provides GPU methods to compute hysteresis curves based on the 
// Preisach model. The code also contains kernels for and for several weighted 
// relay kernels, and utility kernels for comupting individual relays
//

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

///////////////////////
//   CONSTANTS       //
///////////////////////
__device__ __constant__ float PI = 3.141592654f; 
__device__ __constant__ float HB = 6.58211e-16f;
__device__ __constant__ float KB = 8.617332e-5f;
__device__ __constant__ float T  = 320.0f;
__device__ __constant__ float VF = 1e8f;
__device__ __constant__ float G  = 0.4f;
__device__ __constant__ float GS = 4.0f;


//////////////////////////////////
//		ATOMIC OPERATIONS	 	//
//////////////////////////////////
extern "C" {

	__device__ void addToR(float *r, float* rx, float* a, float n) {

		for (int i=0; i<%(XLEN)s;i++){

			atomicAdd(&r[i], rx[i] + n*a[i]);
		}
	}
}

//////////////////////////////////
//	  RELAY DENSITY FUNCTIONS	//
//////////////////////////////////
extern "C" {

	__global__ void RhoConst(float* p){

		int pmx = %(DIM)s*threadIdx.x + threadIdx.y + (%(DIM)s)*(%(DIM)s)* (%(GRID)s*blockIdx.x + blockIdx.y);
		p[pmx]  = 1.0;
	}
}

extern "C" {

	__global__ void RhoElastic(float* p, float* d, float VAL){

		int pmx = %(DIM)s*threadIdx.x + threadIdx.y + (%(DIM)s)*(%(DIM)s)* (%(GRID)s*blockIdx.x + blockIdx.y);
		p[pmx] = (d[pmx] == VAL) ? 1.0: 0.0;

	}
}

extern "C" {

	__global__ void RhoPrandtl(float* p, float* d, float* dw){

		int pmx = %(DIM)s*threadIdx.x + threadIdx.y + (%(DIM)s)*(%(DIM)s)* (%(GRID)s*blockIdx.x + blockIdx.y);
		p[pmx]  =  erf((d[pmx]-dw[0])/(sqrt(float(2.0))*dw[2])) - erf((d[pmx]-dw[1])/(sqrt(float(2.0))*dw[3]));			
	}
}

extern "C" {

	__global__ void RhoErfModel(float* p, float *e, float* d, float* ew, float* dw){

		int pmx = %(DIM)s*threadIdx.x + threadIdx.y + (%(DIM)s)*(%(DIM)s)* (%(GRID)s*blockIdx.x + blockIdx.y);
		p[pmx]  =  (erf((e[pmx]-ew[0])/(sqrt(float(2.0))*ew[2])) - erf((e[pmx]-ew[1])/(sqrt(float(2.0))*ew[3])));
		p[pmx] *=  (erf((d[pmx]-dw[0])/(sqrt(float(2.0))*dw[2])) - erf((d[pmx]-dw[1])/(sqrt(float(2.0))*dw[3])));
		p[pmx] *=  0.25;
	}
}


//////////////////////////
//	  PREISACH KERNEL   //
//////////////////////////
extern "C" {

	__global__ void PreisachASC(float* a, float* e, float*d, float* p, float* r, float n){

		float rx[%(XLEN)s]; 
		int pmx = %(DIM)s*threadIdx.x + threadIdx.y + (%(DIM)s)*(%(DIM)s)* (%(GRID)s*blockIdx.x + blockIdx.y);
		
        for (int i=0; i<%(XLEN)s;i++){

            rx[i] += (a[i] < e[pmx]+d[pmx]) ? float(-p[pmx]): float(p[pmx]);
        }
		
        addToR(r,rx, a, n);
	}   
}

extern "C" {


	__global__ void PreisachDEC(float* a, float* e, float*d, float* p, float* r, float n){

		float rx[%(XLEN)s]; 
		int pmx = %(DIM)s*threadIdx.x + threadIdx.y + (%(DIM)s)*(%(DIM)s)* (%(GRID)s*blockIdx.x + blockIdx.y);
		
        for (int i=0; i<%(XLEN)s;i++){

            rx[i] += (a[i] < e[pmx]-d[pmx]) ? float(-p[pmx]): float(p[pmx]);}
		
        addToR(r,rx, a, n);
	}
}

////////////////////////
//	  RELAY KERNEL	  //
////////////////////////
extern "C" {

	__global__ void RelayASC(float* a, float* r, float e, float d, float n){

		int idx = threadIdx.x;
		r[idx] += (a[idx] < e + d) ? float(-1.0 + n*a[idx]): float(1.0+n*a[idx]);
	}
}

extern "C" {

	__global__ void RelayDEC(float* a, float* r, float e, float d, float n){

		int idx = threadIdx.x;
		r[idx] += (a[idx] < e - d) ? float(-1.0 + n*a[idx]): float(1.0+n*a[idx]);
	}
}

////////////////////////////////
//	  NORMALIZE HYSTERESIS	  //
////////////////////////////////
extern "C" {

	__global__ void Normalize(float* a, float* r, int N){

		int idx = threadIdx.x; 
		float factor = abs(a[0] - a[1])/abs(r[0] - r[N-1]);
		r[idx] *= factor;
		__syncthreads();
		r[idx] -= (r[0]-a[0]);
	}
}
