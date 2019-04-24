################################################
# Script to run henm from henm.py library
################################################
import numpy as np
import henm

# read in 3Nx3N target covariance
dataCovar3N = np.loadtxt("covar3Nx3N.dat")
# read in xyz coordinates of average structure associated with covariance 
avgPos = np.loadtxt("average_structure.dat")
# set the alpha parameter - minimization step size
alpha = 1E-2
# set the number of steps
nSteps = 100
# run the henm procedure - this will generate an initial guess of a uniform force constant matrix
henmHessian = henm.perform_henm(dataCovar3N,avgPos,alpha=alpha,maxIter=nSteps)
# you can also read in a guess
henmHessian = henm.perform_henm(dataCovar3N,avgPos,guessHess=henmHessian,alpha=alpha,maxIter=nSteps)
# save the data file
np.savetxt("henm_hessian_mat.dat",henmHessian)


