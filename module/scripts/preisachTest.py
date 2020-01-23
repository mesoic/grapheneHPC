#!/usr/bin/env python 
#
# Class to calculate preisach kernel with weighted density functions. This kernel 
# was evaluaed as a fit function to ndfit for modelling of graphene field effect 
# devices.
#
# Hysteresis modeling in graphene field effect transistors
#
# Journal of Applied Physics 117, 074501 (2015); https://doi.org/10.1063/1.4913209
import Preisach

# Numerics and plotting
import numpy as np
import matplotlib.pyplot as plt

def tomev(l): return [i*1000 for i in l]

# Methods to test Perisach kernel generation and usage (archive)
if __name__ == "__main__":

    if True:
       
        d = pkl.load(open("./6.pkl"))
        eF, vg = d["eF"], d["vg"]

        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_subplot(111) 
        inset = plt.axes([0.20,0.55,0.32,0.32])

        oP = Preisach(eF, 8)
        oP.grid(1.0,1.0)

        ax1.plot(vg, tomev(eF), "k--")
        oP.evalRelay(-0.00, 0.04,20.0)
        ax1.plot(vg, tomev(oP.r), "r", lw=1)

        inset.plot(tomev(eF), tomev(oP.r), "r",lw=1)
        oP.evalRelay(-0.00, 0.04,0.0)
        inset.plot(tomev(eF), tomev(oP.r), "k--", lw=1)

        inset.set_xlim(-100, 100)
        inset.set_ylim(-120, 120)
        ax1.set_ylim(-120, 100)

        ax1.set_xlabel("Gate Voltage $(V)$")
        ax1.set_ylabel("Fermi Energy $(meV)$")

        inset.set_xlabel("Fermi Energy $(meV)$", fontsize = 18)
        inset.set_ylabel("Fermi Energy $(meV)$", fontsize = 18)
        inset.tick_params(axis='both', which='major', labelsize=18)
        inset.yaxis.set_label_position("right")
        inset.yaxis.set_ticks_position("right")

        ax1.annotate("",
                     xy=(2.1, -27), xycoords='data',
                     xytext=(1.5, -55), textcoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05, width = 1, headwidth = 6),
                     )

        ax1.annotate("",
                     xy=(1.1, 24), xycoords='data',
                     xytext=(1.7, 50), textcoords='data',
                     arrowprops=dict(facecolor='black', shrink=0.05, width = 1, headwidth = 6),
                     )

        inset.annotate("",
                       xy=(25, -32), xycoords='data',
                       xytext=(-25, -65), textcoords='data',
                       arrowprops=dict(facecolor='black', shrink=0.05, width = 1, headwidth = 6),
                       )

        inset.annotate("",
                       xy=(-25, 24), xycoords='data',
                       xytext=(25, 60), textcoords='data',
                       arrowprops=dict(facecolor='black', shrink=0.05, width = 1, headwidth = 6),
                       )
        plt.savefig("functionalRelay.eps",bbox_inches='tight')    
        plt.savefig("functionalRelay.png",bbox_inches='tight')    
        plt.tight_layout()
        plt.show()

    if False: 
        d = pkl.load(open("./testwave2.pkl"))
        vg, eF = d["vg"], d["eF"]

        
        oP = Preisach(eF, 16, vg)
        oP.grid(1.0,1.0)

        E = [-0.100, 0.100, 0.060, 0.060]
        D = [ 0.050, 0.100, 0.050, 0.050]


        #oP.rhoconst()
        oP.rho(E,D)
        oP.eval(5.0)
               
        plt.figure(2)
        plt.plot(vg,oP.eF)
        plt.plot(vg,oP.r)
        plt.show()
        
    if False:

        if False: 

            eF  = np.linspace(1.0,-1.0,512).astype(np.float32)
            _eF = np.linspace(-1.0,1.0,512).astype(np.float32) 
            eF = np.concatenate([eF, _eF])

            oP = Preisach(eF, 8)
            oP.grid(1.0,1.0)
            
            # Constant (Pure Preisach)
            oP.rhoconst()
            oP.eval()
            plt.plot(eF,oP.r, 'k')

            # Elastic-Plastic
            oP.rhoElastic(0.25)
            oP.eval()
            plt.plot(eF,oP.r, 'r')
            
            # Prandtl in Width
            D = [0.050, 0.15, 0.25, 0.25]
            oP.rhoPrandtl(D)
            oP.eval()
            plt.plot(eF,oP.r,'b')

            # Preisach Full Rho
            #E = [-0.100, 0.100, 0.10, 0.10]
            #D = [ 0.000, 0.150, 0.25, 0.25]
            #oP.rho(E,D)
            #oP.eval()
            #plt.plot(eF,oP.r,'g')

            #plt.plot(eF, eF, 'k', lw = 2 )

            plt.ylim(-1,1)
            plt.show()
            

        # KERNEL TESTING
        if True:
            eF  = np.linspace(1.0,-1.0,128).astype(np.float32)
            _eF = np.linspace(-1.0,1.0,128).astype(np.float32) 
            eF = np.concatenate([eF, _eF])

            for ORDER in [1,2,4,8,16,32,64,128]:
                oP = Preisach(eF, ORDER)
                oP.grid(1,1)
                oP.rhoconst()
                oP.eval()
                plt.plot(eF,oP.r,'r')
            plt.plot(eF,oP.r,'k')
            plt.plot(eF, eF,'k:') 
            plt.ylim(-1,1)
            plt.show()

        # KERNEL TESTING
        if False:
            eF  = np.linspace(1.0,-1.0,128).astype(np.float32)
            _eF = np.linspace(-1.0,1.0,128).astype(np.float32) 
            eF = np.concatenate([eF, _eF])

            oP = Preisach(eF, 1) 
            oP.grid(1,1) 
            oP.rhoconst()
            oP.eval()
            plt.plot(eF, oP.r) 
            plt.show()

        # Tesing ERF weighting 
        if False:
            E = [  -0.500 , 0.500, 0.05, 0.15]
            D = [   0.100 , 0.700, 0.15, 0.05]

            oP = Preisach(eF, 16) 
            oP.grid(1.0,1.0)
            oP.rho(E,D)
            oP.eval()
            oP.plotresult()
            oP.rhoconst()
            oP.eval()
            oP.plotresult('b')
            plt.show()
            
        # TESTING PLOT RHO
        if False:
            E = [  -0.080 , 0.030, 0.10, 0.15]
            D = [   0.100 , 1.500, 0.15, 0.05]

            oP = Preisach(eF, 8) 
            oP.grid(1.0,1.0)
            oP.rho(E,D)
            oP.plotrho()
            plt.show()
               
        # Testing carrier density calculator
        if False:
            NPOINTS = 64
            eF   = np.linspace(-0.5,0.5,NPOINTS).astype(np.float32)
            oP = Preisach(eF, 16)
            oP.grid(1,1)
            oP.rhoconst()
            tic = datetime.now()
            n,p = oP.carriers(eF)
            toc = datetime.now()

            print "Integrated for (n) and (p) on %d points in %dus"%(NPOINTS, (toc-tic).microseconds)

            plt.figure(1)
            plt.plot(eF)
            plt.figure(2)
            plt.semilogy(eF, n)
            plt.semilogy(eF, p, 'r')
            plt.xlim(-0.5,0.5)
            plt.xlabel("Fermi Energy (eV)")
            plt.ylabel("Carrier Density (eV)")
            plt.show()
