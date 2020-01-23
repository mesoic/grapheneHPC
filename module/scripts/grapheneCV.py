#!/usr/bin/env python 
#
# Characterization and physical modeling of MOS capacitors in epitaxial 
# graphene monolayers and bilayers on 6H-SiC
#
# AIP Advances 6, 085010 (2016); https://doi.org/10.1063/1.4961361
#
import cPickle as pkl

# Import graphene kernel
import grapheneKernel as k

# Numerics and matplotlib
import numpy as np
import ndfit as ndf 
import matplotlib.pyplot as plt 


# Capacitance
def solve_c(eF, eG, deF, Dit, COX, F, M,  DEPTH = 1000):

	return oP.CAPACITANCE(eF, eG, deF, mp(F, Dit), COX, M, DEPTH)


# Fermi energy waves
def solve_eF(V, vD, Dit, eG, COX, TEMP, F, M, FLYWHEEL=0.1):

	eF = 0.1*np.sign(np.asarray(V))*np.sqrt(abs(np.asarray(V)))
	return oP.SOLVE_EF( V, vD, mp(F, Dit), eF, eG, COX, TEMP, M, FLYWHEEL)
  

# Voltage 
def get_vD(vG, Clf): 

	vD =  min( enumerate(zip(Clf, vG)), key=lambda pt: pt[1][0])
	return [ vD[1][1], vD[0]]


###########################################
#			 FITTING ENGINE				  #
###########################################
#
#  _DATA  = [DitV["Vb"],  DitV["Dit"], DitV["C%d"%index ]]
#  params = [de[0], de[1], FACTOR]
#  consts = [vDf, eG[0], eG[1], COX, TEMP, M]
#

def fitfunc(dat, p, c):
	eF_fit = solve_eF(dat[0], c[0], dat[1], [c[1], c[2]], c[3], c[4], p[2], c[5], FLYWHEEL=1.0)
	C_fit  = solve_c(eF_fit, [c[1],c[2]], [p[0],p[1]], dat[1], c[3], p[2], c[5], DEPTH = 2100)
	return top(C_fit*AREA)
						  

def errfunc(dat,p, c): 
	return [(fit-dat) for fit,dat in zip(fitfunc(dat,p,c), top(dat[2]))]


# Main method: This method invokes the graphene kernel class inside of 
# ndfit to optimize surface potential energy fluctuation statistics 
# for accurate bandgap smoothing in graphene monolayers and bilayers
if __name__ == "__main__":

	# Unit conversion
	def tom(vec): return [1e03*_ for _ in vec]
	def tou(vec): return [1e06*_ for _ in vec]
	def top(vec): return [1e12*_ for _ in vec]
	def mp(F, vec): return [F*_ + 0.00e12  for _ in vec]


	# Initialize CUDA kernel
	global oP 
	oP	=  k.grapheneKernel()

	# Physical constants
	hbar =  4.1356e-15  #eV s
	vf	 =  1.0000e+08  #cm/s
	e	 =  1.6020e-19
	AREA =  (100e-4*100e-4)

	index  = 4
	COX	   = (35.40e-12)/AREA  ## Per Unit Area 
	DitV   = pkl.load( open("ML/Dit_ML.p", "rb") ) 
	vDf	   = get_vD(DitV["Vb"][:100], DitV["C4"][:100])
	vDr	   = get_vD(DitV["Vb"][100:], DitV["C4"][100:])

	# Bandgap for bilayers. Note that eG < 0 returns standard DOS
	eG	   = [0.1300, 0.0950]  ## [eG/2, eGs] error function rolloff
	deF	   = [0.0938, 0.1898]  ## [deF , deS]  potential fluctuations 
	TEMP   = 77.0			   ## TEMPERATURE
	FACTOR = 1.0			   ## FACTOR FOR Dit 
	M	   = 0.8			   ## Monolayer/Bilayer Ratio

	print vDf

	# For calling fit function 
	_DATA  = [DitV["Vb"],  DitV["Dit"], DitV["C%d"%index ]]
	params = [deF[0], deF[1], FACTOR]
	consts = [vDf, eG[0], eG[1], COX, TEMP, M]
	step   = [0.001, 0.001, 0.001]

	if False:

		ndf.maxdepth(30)
		ndf.convergence(0.001)
		ndf.throttle_factor(0.1)
		NDF = ndf.run(fitfunc, errfunc, _DATA, params, consts, step, mode="short", throttle=True)
		p   = NDF.getresult()[1]
		print p
		params = p

	C_fit  = fitfunc(_DATA, params, consts)
	plt.plot(DitV["Vb"], C_fit,'k')
	plt.plot(DitV["Vb"], top(DitV["C%d"%index]))
	plt.show()
	
	# Monolayer Fit Parameters
	# Use C4 for vD
	if False:
		M, COX = 0.8, 35.60e-12
		p0,eG0 = [0.1780, 0.1698, 8.00],[0.135, 0.095] 
		p1,eG1 = [0.1598, 0.1898, 7.00],[0.137, 0.092] 
		p2,eG2 = [0.1590, 0.1608, 4.69],[0.135, 0.095]
		p3,eG3 = [0.1428, 0.1558, 3.69],[0.137, 0.092]
		p4,eG4 = [0.1128, 0.1558, 1.10],[0.137, 0.092]
		p5,eG5 = [0.0780, 0.1400,-2.40],[0.137, 0.092]
		FIT	= [p0, p1, p2, p3, p4, p5]
		EG	 = [eG0, eG1, eG2, eG3, eG4, eG5]
		pkl.dump([FIT, EG, M, COX], open("CV_params_ML.p", "wb"))

	# Bilayer Fit Parameters
	if False: 
		M, COX  = 0.10, 33.00e-12
		p5, eG5 = [0.080,0.105,0.00],[0.123, 0.065] ##C5
		p4, eG4 = [0.080,0.105,3.00],[0.130, 0.080] ##C4 
		p3, eG3 = [0.090,0.110,5.00],[0.130, 0.080] ##C3
		p2, eG2 = [0.091,0.105,5.15],[0.130, 0.080] ##C2
		p1, eG1 = [0.091,0.105,6.80],[0.130, 0.080] ##C1
		p0, eG0 = [0.091,0.105,6.90],[0.130, 0.080] ##C0
		FIT	 = [p0, p1, p2, p3, p4, p5]
		EG	  = [eG0, eG1, eG2, eG3, eG4, eG5]
		pkl.dump([FIT, EG, M, COX], open("CV_params_BL.p", "wb"))
