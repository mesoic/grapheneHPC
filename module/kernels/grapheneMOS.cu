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
// Graphene CV Modeling Kernel
//
// Compile with: nvcc -c -arch=sm_20 <filename>.cu	
// 
// Compilation will create CV_kernel.cubin which provides the
// functions below. The .cubin can then be imported into 
// Python(PyCUDA) or C/C++ code.	
//
//////////////////////////////////////////////////////////////////////////
//
// This kernel provides GPU methods to calculate the monolayer and bilayer 
// DOS, eF, Capacitance, and other related quantities. Dit and surface 
// potential fluctuations are also included. 
//
// Characterization and physical modeling of MOS capacitors in epitaxial 
// graphene monolayers and bilayers on 6H-SiC
//
// AIP Advances 6, 085010 (2016); https://doi.org/10.1063/1.4961361
// 

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

///////////////////////
//	 CONSTANTS		 //
///////////////////////

__device__ __constant__ float PI = 3.141592654f; 
__device__ __constant__ float HB = 6.58211e-16f;
__device__ __constant__ float KB = 8.617332e-5f;
__device__ __constant__ float VF = 1.000000e+8f;
__device__ __constant__ float G	= 0.400f;
__device__ __constant__ float GS = 4.000f;
__device__ __constant__ float e	= 1.602e-19f;
__device__ __constant__ float R	= 1.414213562f;

//////////////////////////////////
//		ATOMIC OPERATIONS	 	//
//////////////////////////////////

extern "C" {

	__device__	void addToC(float *r, float* rx, float* a, float n, int INT) {
		
		for (int i=0; i<INT;i ++){
		
			atomicAdd(&r[i], rx[i] + n*a[i]);
		}
	}
}

//////////////////////////////////////////
//	 CALCULATION OF DOS/CARRIERS  		//
//////////////////////////////////////////

// Monolayer
extern "C"{

	__device__ float mlDOS_PW(float eF){

		return (GS/(2.0*PI))*(1.0/(HB*HB*VF*VF))*abs(eF); 
	
	}
}

extern "C" {

	__device__ float mlCarriers_PW(float ef, float T){
			
		int INT = 1024;
		float n = 0.0;
		float p = 0.0;

		for(int i = 0; i<INT; i++){

			n += mlDOS_PW(float(i)/float(INT))*(1.0/(1.0 + exp( ((float(i)/float(INT)) -ef )/(KB*T))))*(1.0/INT);
			p += mlDOS_PW(float(i)/float(INT))*(1.0/(1.0 + exp( ef + (float(i)/float(INT)) )/(KB*T)))*(1.0/INT);
		}
	
		return n-p;
	}
}

// Bilayer. DOS includes energy gap model 
extern "C" { 
	
	__device__ float blDOS_PW(float eF, float eG, float eGs){
	
		if (eG < 0.0){ return (GS/(2.0*PI))*(1.0/(HB*HB*VF*VF))*(abs(eF) + (G/2.0));}
	 
		double dFp; 
		double dFm;
	
		dFp = float( 0.5* (1 -	erf (	double(	(eF+eG)/(R*eGs) ) )));
		dFm = 1.0 - float( 0.5* (1 -	erf ( double (	(eF-eG)/(R*eGs) ) )));
	
		return (float(dFm)+float(dFp)) * (GS/(2.0*PI)) * (1.0/(HB*HB*VF*VF))*(abs(eF) + (G/2.0));
	}
} 

extern "C" {
	__device__ float blCarriers_PW(float ef, float eG, float eGs, float T){
		
		int INT = 1024;
		float n = 0.0;
		float p = 0.0;

		for(int i = 0; i<INT; i++){

			n += blDOS_PW(float(i)/float(INT), eG, eGs) * (1.0/(1.0 + exp((	(float(i)/float(INT)) -ef	)/(KB*T))))*(1.0/INT);
			p += blDOS_PW(float(i)/float(INT), eG, eGs) * (1.0/(1.0 + exp((	ef + (float(i)/float(INT)) )/(KB*T))))*(1.0/INT);
		}
		
		return n-p;
	}
}

//////////////////////////////////////////////////
//	 MONTE CARLO SIMULATION of CAPACITANCE		//
//////////////////////////////////////////////////

extern "C" {

	__global__ void gen_rand(float* eF, float* eFr, float deF, unsigned long SEED){
		const int tid = threadIdx.x;
	
		curandState state;
		curand_init(SEED, tid, 0, &state);

		eF[tid]	= eF[tid];
		eFr[tid] = deF*curand_normal(&state) + eF[tid];
	}
}

extern "C" {

	__global__ void gen_capMC(float* eF, float* eFr, float* eG, float* deF, float* Dit, 
								float*C, float COX, float M, int DEPTH, unsigned long SEED){
	
		const int tid = threadIdx.x;
		curandState state;
		curand_init(SEED, tid, 0, &state);

		float num;
		float den;

		for (int i = 0; i<DEPTH; i++){ 

			eFr[tid] = deF[0]*( exp(-1* (eF[tid]*eF[tid])/(2*deF[1]*deF[1])) )*curand_normal(&state) + eF[tid]; 

			num	= (e*COX * ( M*mlDOS_PW(eFr[tid]) + (1.0-M)*blDOS_PW(eFr[tid], eG[0], eG[1]) + Dit[tid]));	
			den	= (COX + e*M*mlDOS_PW(eFr[tid]) + e*(1.0-M)*blDOS_PW(eFr[tid], eG[0], eG[1]) + e*Dit[tid]);
			C[tid]	+= num/den;
		} 

		C[tid] = C[tid]/float(DEPTH);
	}
}

///////////////////////////////////////////
//			 NEWTONS METHOD				 // 
///////////////////////////////////////////

extern "C"{

	__device__ float integrate(int v_i, int vD_i, float* Dit, float DELTA){
	
		float _int = 0;
	
		// Some bad coding to take care of return sweep
		if (v_i == 200){
			v_i = 0;
		}
		
		if (v_i > 100){
			v_i -= 2*(v_i%100);
		}
	
		// Otherwise just integrate like normal
		if (v_i < vD_i){

			for(int i = v_i; i < vD_i; i++){

				_int += Dit[i]*DELTA ; 
			}
		}	
		
		else{ 

			for(int i = vD_i; i < v_i; i++){

				_int -= Dit[i]*DELTA ; 
			} 
		}	
		
		return _int;
	}
}

extern "C"{

	__global__ void solve_eF(float* v, float* vD, float* Dit, float* eF, float* eG, 
				 			 float COX, float T, float M, float FLYWHEEL){

		const int tid = threadIdx.x;
		const float DELTA = v[1]-v[0];

		float lhs = v[tid] - vD[0] - ((e/COX)*integrate(tid, int(vD[1]), Dit, DELTA)); 
		float rhs = ( eF[tid] + M*(e/COX)*mlCarriers_PW(eF[tid], T)	+ (1-M)*(e/COX)*blCarriers_PW(eF[tid], eG[0], eG[1], T) );
		float dhs;

		if (tid == int(vD[1])){
			eF[tid] = 0.0; 
			return;
		}
		
		if (tid - 2*(tid%100) == int(vD[1])){
			eF[tid] = 0.0; 
			return;
		}

		for (int i = 0; i < 100; i++){ 
			
			float conv = (lhs - rhs);
			
			if (abs(conv) < 1e-6){ 
				return; 
			}
			
			else{
			
				rhs = (	eF[tid] + M*(e/COX)*mlCarriers_PW(eF[tid], T) + (1-M)*(e/COX)*blCarriers_PW(eF[tid], eG[0], eG[1], T) );
				dhs = (	1 + M*(e/COX)*mlDOS_PW(eF[tid]) + (1-M)*(e/COX)*blDOS_PW(eF[tid], eG[0], eG[1]) );
				eF[tid] += FLYWHEEL*(lhs - rhs)/(dhs);
			}
		}
		return;
	}
}

//////////////////////////////////////////
//		VECTORWISE CARRIERS/DOS			//
//////////////////////////////////////////

extern "C"{

	__global__ void mlDOS_vec(float* eF, float* RO){ 
	
		const int tid = threadIdx.x;

		RO[tid] = mlDOS_PW(eF[tid]);
		return;
	}
}

extern "C"{

	__global__ void blDOS_vec(float* eF, float* eG, float* RO){ 

		const int tid = threadIdx.x;

		RO[tid] = blDOS_PW(eF[tid], eG[0], eG[1]);

		return;
	}
}

extern "C"{
	__global__ void mlCarriers_vec(float* eF,	float* n, float* p, float t){ 

		const int tid	 = threadIdx.x; 

		int INT = 1024;

		for(int i = 0; i<INT; i++){
			n[tid] += mlDOS_PW(float(i)/float(INT))*(1.0 / (1.0 + exp( ((float(i)/float(INT)) -eF[tid]) /(KB*t))))*(1.0/INT);
		}

		for(int i = 0; i<INT; i++){
			p[tid] += mlDOS_PW(float(i)/float(INT))*(1.0 / (1.0 + exp( ((float(i)/float(INT)) +eF[tid]) /(KB*t))))*(1.0/INT);
		}

		return;
	}
}

extern "C"{

	__global__ void blCarriers_vec(float* eF,	float* eG, float* n, float* p, float t){ 

		const int tid = threadIdx.x; 
	
		int INT = 1024;
		for(int i = 0; i<INT; i++){
			n[tid] += blDOS_PW(float(i)/float(INT), eG[0], eG[1])*(1.0 / (1.0 + exp( ((float(i)/float(INT)) -eF[tid]) /(KB*t))))*(1.0/INT);
		}
		for(int i = 0; i<INT; i++){
			p[tid] += blDOS_PW(float(i)/float(INT), eG[0], eG[1])*(1.0 / (1.0 + exp( ((float(i)/float(INT)) +eF[tid]) /(KB*t))))*(1.0/INT);
		}
	
		return;
	}
}
