#!/usr/bin/env python 
#
# Class to calculate preisach kernel with weighted density functions. This kernel 
# was evaluaed as a fit function to ndfit for modelling of graphene field effect 
# devices.
#
# Hysteresis modeling in graphene field effect transistors
#
# Journal of Applied Physics 117, 074501 (2015); https://doi.org/10.1063/1.4913209

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Utilities
import copy
from datetime import datetime

# Numerics and plotting
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

# Preisach kernel class (archived)
class preisachKernel:
   
    def __init__(self, eF, NGRID, vg=None):

        # Fermi energy
        self.eF = np.array(eF).astype(np.float32) 
        self.vg = list(vg) if vg is not None else None
        
        # Break fermi energy into ascending and descending 
        # branches for kernel evaluation
        self._eval = self.getMonotonic(self.eF)
        
        # Block and grid for GPU
        self.nblock = 16 
        self.NGRID = NGRID

        # Read kernel template file
        f = open("./cuda/graphene.cubin", "rb")
        kernel_code_template = f.read()
        
        # Preprocessor
        kernel_code = kernel_code_template %{'DIM' : self.nblock,'GRID': self.NGRID, 'XLEN': 108}

        # Load kernel source module
        self.mod = SourceModule(kernel_code)


    #####################################     
    #   RELAY GRID AND DOMAIN
    #  
    
    
    # For get monotonic 
    def chunks(self, l, n):
        for i in xrange(0, len(l), n):
            yield l[i:i+n]
    

    # Create grid for kernel evaluation       
    def grid(self, a = 1.0, b = 1.0):
       
        de   = b*(max(self.eF)-min(self.eF)) 
       
        _e = a*np.linspace( min(self.eF), max(self.eF), self.NGRID*self.nblock)
        _d = np.linspace( 0.0, de, self.NGRID*self.nblock)
        
        e,d  = np.meshgrid(_e, _d)
        self.e = e.astype(np.float32)
        self.d = d.astype(np.float32)


    # Function to break list into monotonically increasing/decreasing chunks. 
    # This is something we only need to do exactly ONCE.
    def getMonotonic(self,eF, size):
        _eval, tmp,  S = [],[], eF[1] - eF[0]
        for i in range(0, len(eF)-1):
            if S > 0:
                if self.eF[i] >= self.eF[i-1]: 
                    tmp.append(self.eF[i])
                else:
                    _eval.append((tmp,True)) 
                    tmp = [eF[i]]
                    S = eF[i+1] - eF[i]
            else:
                if self.eF[i] < self.eF[i-1]: 
                    tmp.append(self.eF[i])
                else:
                    _eval.append((tmp,False)) 
                    tmp = [eF[i]]
                    S = eF[i+1] - eF[i]
                    
        # Take care of final subsequence
        if  (eF[-1] - eF[-2]) > 0:
            tmp.append(eF[-1])
            _eval.append((tmp,True))
        else:
            tmp.append(eF[-1])
            _eval.append((tmp,False))

        for i,block in enumerate(_eval):
            _eval[i] = (list(self.chunks(block[0], size) ), block[1])
       
        if _eval[0][0]==[]:
            return _eval[1:]
        else:
            return _eval


    #####################################     
    #   RELAY DENSITY FUNCTIONS
    #        

    # Relay density function
    def rhoErfModel(self, _E, _D):

        self.p = np.zeros_like(self.e).astype(np.float32)
        E = np.array(_E).astype(np.float32)
        D = np.array(_D).astype(np.float32)

        p_gpu  = cuda.mem_alloc(self.p.nbytes) 
        e_gpu  = cuda.mem_alloc(self.e.nbytes) 
        d_gpu  = cuda.mem_alloc(self.d.nbytes)
        E_gpu  = cuda.mem_alloc(E.nbytes)
        D_gpu  = cuda.mem_alloc(D.nbytes)

        cuda.memcpy_htod(p_gpu, self.p)
        cuda.memcpy_htod(e_gpu, self.e)
        cuda.memcpy_htod(d_gpu, self.d)
        cuda.memcpy_htod(E_gpu, E)
        cuda.memcpy_htod(D_gpu, D)
        
        RHO  = self.mod.get_function("Rho")
        RHO(p_gpu, e_gpu, d_gpu, E_gpu, D_gpu, 
            block=(self.nblock, self.nblock, 1), grid=(self.NGRID,self.NGRID))

        cuda.memcpy_dtoh(self.p, p_gpu)

    def rhoConst(self):
        
        self.p = np.zeros_like(self.e).astype(np.float32)

        p_gpu  = cuda.mem_alloc(self.p.nbytes) 
        
        cuda.memcpy_htod(p_gpu, self.p)

        RHO  = self.mod.get_function("RhoConst")
        RHO(p_gpu, block=(self.nblock, self.nblock, 1), grid=(self.NGRID,self.NGRID))
        
        cuda.memcpy_dtoh(self.p, p_gpu)

    def rhoElastic(self, val):

        self.p = np.zeros_like(0.5*self.d).astype(np.float32) 
        val    = min(list(self.d[:,0]), key=lambda i: abs(i - val))

        p_gpu  = cuda.mem_alloc(self.p.nbytes) 
        d_gpu  = cuda.mem_alloc(self.d.nbytes)
        
        cuda.memcpy_htod(p_gpu, self.p)
        cuda.memcpy_htod(d_gpu, self.d)

        RHO  = self.mod.get_function("RhoElastic")
        RHO(p_gpu, d_gpu, val.astype(np.float32), 
            block=(self.nblock, self.nblock, 1), grid=(self.NGRID,self.NGRID))

        cuda.memcpy_dtoh(self.p, p_gpu)
        
    def rhoPrandtl(self, _D):

        self.p = np.zeros_like(self.e).astype(np.float32)
        D = np.array(_D).astype(np.float32)
   
        p_gpu  = cuda.mem_alloc(self.p.nbytes) 
        d_gpu  = cuda.mem_alloc(self.d.nbytes)
        D_gpu  = cuda.mem_alloc(D.nbytes)

        cuda.memcpy_htod(p_gpu, self.p)
        cuda.memcpy_htod(d_gpu, self.d)
        cuda.memcpy_htod(D_gpu, D)

        RHO  = self.mod.get_function("RhoPrandtl")
        RHO(p_gpu, d_gpu, D_gpu, block=(self.nblock, self.nblock, 1), grid=(self.NGRID,self.NGRID))
        
        cuda.memcpy_dtoh(self.p, p_gpu)
        

    #####################################     
    #  RELAY EVALUATION FUNCTIONS
    #   

    def evalRelay(self,E,D, n=0.0):
        
        tic = datetime.now()
        ############################################################################
        self.nn, self.pp,self.r, tmp  = [],[],[],[]
        for block in self._eval:
            if block[1] == True:
                for b in block[0]:
                    tmp += self.evalRelayASC(np.array(b).astype(np.float32),E,D,n)
                self.r  += self.evalScale([min(self.eF),max(self.eF)], tmp)
                tmp = []
            if block[1] == False:
                for b in block[0]:
                    tmp += self.evalRelayDEC(np.array(b).astype(np.float32),E,D,n)
                self.r +=  self.evalScale([max(self.eF),min(self.eF)], tmp)
                tmp = []

        #self.nn, self.pp = self.carriers(self.r) 
        ###########################################################################
        toc = datetime.now()
        t   = toc - tic
        
        #print("Evaluation Cycle: %d Hysterions on %d points in %d us"
        # %( (self.nblock*self.NGRID)**2, len(self.eF), t.microseconds) )

    # Ascending Branch        
    def evalRelayASC(self, eF, E, D, n):

        LEN = np.int32(len(eF))
        E   = np.float32(E)
        D   = np.float32(D)
        n   = np.float32(n)
        r  = np.zeros_like(eF).astype(np.float32) 
        
        eF_gpu = cuda.mem_alloc(eF.nbytes)
        r_gpu  = cuda.mem_alloc(r.nbytes)
        
        cuda.memcpy_htod(eF_gpu, eF)
        cuda.memcpy_htod(r_gpu , r)
        
        RELAY_ASC  = self.mod.get_function("RelayASC")
        RELAY_ASC(eF_gpu, r_gpu, E, D, n, block=(int(LEN), 1, 1))
        
        cuda.memcpy_dtoh(r, r_gpu)
        return list(r)

    # Descending Branch
    def evalRelayDEC(self, eF, E, D, n):
      
        LEN = np.int32(len(eF))
        E   = np.float32(E)
        D   = np.float32(D)
        n   = np.float32(n)
        r  = np.zeros_like(eF).astype(np.float32) 
      
        eF_gpu = cuda.mem_alloc(eF.nbytes)
        r_gpu  = cuda.mem_alloc(r.nbytes)
      
        cuda.memcpy_htod(eF_gpu, eF)
        cuda.memcpy_htod(r_gpu , r)
      
        RELAY_DEC  = self.mod.get_function("RelayDEC")
        RELAY_DEC(eF_gpu, r_gpu, E, D, n, block=(int(LEN), 1, 1))
      
        cuda.memcpy_dtoh(r, r_gpu)
        return list(r)


    #####################################     
    #   KERNEL EVALUATION FUNCTIONS
    #   

    def eval(self, n=0.0):
        
        tic = datetime.now()
        ###########################################
        self.nn, self.pp,self.r, tmp  = [],[],[],[]
        for block in self._eval:
            if block[1] == True:
                for b in block[0]:
                    tmp += self.evalASC(np.array(b).astype(np.float32),n)
                self.r  += self.evalScale([min(self.eF),max(self.eF)], tmp)
                tmp = []
            if block[1] == False:
                for b in block[0]:
                    tmp += self.evalDEC(np.array(b).astype(np.float32),n)
                self.r +=  self.evalScale([max(self.eF),min(self.eF)], tmp)
                tmp = []

        self.nn, self.pp = self.carriers(self.r) 
        ###########################################
        toc = datetime.now()
        t   = toc - tic

        #print("Evaluation Cycle: %d Hysterions on %d points in %d us"
        #%( (self.nblock*self.NGRID)**2, len(self.eF), t.microseconds)
        
    # Ascending branch
    def evalASC(self,eF,n):
      
        LEN = np.int32(len(eF))
        n   = np.float32(n)
        r = np.zeros_like(eF).astype(np.float32) 
      
        eF_gpu = cuda.mem_alloc(eF.nbytes)
        r_gpu  = cuda.mem_alloc(r.nbytes)
        d_gpu  = cuda.mem_alloc(self.d.nbytes)
        e_gpu  = cuda.mem_alloc(self.e.nbytes)
        p_gpu  = cuda.mem_alloc(self.p.nbytes)
      
        cuda.memcpy_htod(eF_gpu, eF)
        cuda.memcpy_htod(r_gpu , r)
        cuda.memcpy_htod(e_gpu , self.e)
        cuda.memcpy_htod(d_gpu , self.d)
        cuda.memcpy_htod(p_gpu , self.p)


        tic = datetime.now()
        ############################################
        PREISACH_ASC  = self.mod.get_function("PreisachASC")
        PREISACH_ASC(eF_gpu, e_gpu, d_gpu, p_gpu, r_gpu,n,
            lock=(self.nblock, self.nblock, 1), grid=(self.NGRID,self.NGRID))
        cuda.memcpy_dtoh(r, r_gpu)
        ############################################
        toc = datetime.now()
        t   =  toc-tic
        
        #print("%d Hysterions (ASC) evaluated on %d points in %dus"
        # %( (self.nblock*self.NGRID)**2,int(LEN),t.microseconds))
        return list(r)
                
    # Descencing Branch
    def evalDEC(self, eF,n):
        
        LEN = np.int32(len(eF))
        n   = np.float32(n)
        r  = np.zeros_like(eF).astype(np.float32) 
        
        eF_gpu = cuda.mem_alloc(eF.nbytes)
        r_gpu  = cuda.mem_alloc(r.nbytes)
        d_gpu  = cuda.mem_alloc(self.d.nbytes)
        e_gpu  = cuda.mem_alloc(self.e.nbytes)
        p_gpu  = cuda.mem_alloc(self.p.nbytes)
        
        cuda.memcpy_htod(eF_gpu, eF)
        cuda.memcpy_htod(r_gpu , r)
        cuda.memcpy_htod(e_gpu , self.e)
        cuda.memcpy_htod(d_gpu , self.d)
        cuda.memcpy_htod(p_gpu , self.p)

        tic = datetime.now()
        ############################################
        PREISACH_DEC  = self.mod.get_function("PreisachDEC")
        PREISACH_DEC(eF_gpu, e_gpu, d_gpu, p_gpu, r_gpu, n, 
            block=(self.nblock, self.nblock, 1), grid=(self.NGRID,self.NGRID))
        cuda.memcpy_dtoh(r, r_gpu)
        ############################################
        toc = datetime.now()
        t   =  toc-tic

        #print("%d Hysterions (DEC) evaluated on %d points in %dus"
        # %( (self.nblock*self.NGRID)**2,int(LEN),t.microseconds)
        return list(r)


    # Normalize hysterestis to number of points
    def evalScale(self, eX, r):

        LEN = np.int32(len(r))
        r0 = np.asarray(r).astype(np.float32)
        eX = np.asarray(eX).astype(np.float32)

        r0_gpu = cuda.mem_alloc(r0.nbytes)
        eX_gpu = cuda.mem_alloc(eX.nbytes)

        cuda.memcpy_htod(eX_gpu , eX)
        cuda.memcpy_htod(r0_gpu , r0)

        SCALE = self.mod.get_function("Scale")
        SCALE(eX_gpu, r0_gpu, LEN ,block=(int(LEN) , 1 , 1 ) )

        cuda.memcpy_dtoh(r0, r0_gpu)
        return list(r0)


    #####################################     
    #   POSTPROCESSING : PLOTTING
    #         

    # Plot optimized relay density function    
    def plotrho(self, save=False):
   
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.e, self.d, self.p, 
                        rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel( r'Energy $(eV)$', fontsize=16)
        ax.set_ylabel( r'Width $(eV)$', fontsize=16)
        ax.set_zlabel( r'$\rho (\epsilon, \epsilon_\delta)$', fontsize=16)
    
        if save:
            fig.savefig('rho.eps', bbox_inches='tight', pad_inches=0)

    # Plot kernel result     
    def plotresult(self, c = 'r'):
    
        fig = plt.figure(2)
        plt.plot(self.eF,self.r, c)
        plt.plot(self.eF,self.eF, 'k:')
        plt.ylim(min(self.r),max(self.r))
