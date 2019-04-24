#####################################################
#  hetero Elastic Network Model python library 
#####################################################
import numpy as np
import sys
# use numba to speed things up
from numba import jit

# parent routine to perform the hENM procedure
@jit
def perform_henm(targetCovar,avgPos,guessHess = [],alpha=0.1,maxIter=10, thresh=1e-4):
    # targetCovar - 3Nx3N target covariance
    # avgPos - Nx3 array of average node positions
    # guessHess - optional 3Nx3N guess Hessian 
    # alpha - scalar float value that dicates force constant correction step size
    # maxIter - scalar integer value that dicatates maximum number of steps to perform
    # thresh - scalar float value that dictates convergence threshold
    N3 = targetCovar.shape[0]
    N = N3//3
    # guess a Hessian if none is passed
    if guessHess == []:
        # create initial Hessian as all 10s
        hessian = -10.0*np.ones((N,N),dtype=np.float64)
        #hessian = reachHessian
        for i in range(N):
            hessian[i,i] = 0.0
            hessian[i,i] = -np.sum(hessian[i,:])
    else:
        hessian = guessHess
    # compute pairwise displacement vectors
    dispVecs = pairwise_disp_vecs(avgPos)
    # convert target covar to target residual
    targetRes = residual_from_covar(targetCovar,dispVecs)
    # iterate
    step = 0
    conv = "FALSE"
    while step < maxIter and conv == "FALSE":
        # project hessian in 3Nx3N
        hessian3N = project_NxN_to_3Nx3N(hessian,dispVecs)
        # invert hessian - using psuedo inverse
        covar3N = np.linalg.pinv(hessian3N,rcond=1e-12)*0.6  # multiply by thermal energy
        # take difference of tensor covars and project into separation vector
        covarDiff = compute_henm_diff(covar3N,targetRes,dispVecs)
        # Check if conveged
        dev = np.linalg.norm(covarDiff)
        if dev < thresh:
            conv = "TRUE"
        else: # update Hessian
            hessian = update_hessian(hessian,covarDiff,alpha)
        step += 1
        print(step, dev)
        sys.stdout.flush()

    return hessian

###############################################################
#  Subroutines called by the parent routine
###############################################################

# generate an NxN matrix from a 3Nx3N matrix by taking trace of 3x3 sub-tensors
@jit
def NxN_from_3Nx3N(matrix):
    N3 = matrix.shape[0]
    N = N3//3
    k = np.empty((N,N),dtype=np.float64)
    for i in range(N):
        iIndex = [i*3,i*3+1,i*3+2]
        for j in range(N):
            jIndex = [j*3,j*3+1,j*3+2]
            k[i,j] = np.trace(matrix[i*3:i*3+3,j*3:j*3+3])
    return k    

# generate a 3Nx3N matrix from an NxN matrix and the displacement vectors between nodes
# NOTE: this routine is specific to Hessian/graph Laplacians because of the way it populates the diagonal elements
@jit
def project_NxN_to_3Nx3N(hessian,dispVecs):
    N = hessian.shape[0]
    N3 = N*3
    newMat = np.zeros((N3,N3),dtype=np.float64)
    for i in range(N):
        for j in range(i):
            diff = dispVecs[i,j,:]
            newMat[i*3:i*3+3,j*3:j*3+3] = hessian[i,j] * np.outer(diff,diff)
            # symmetrize
            newMat[j*3:j*3+3,i*3:i*3+3] = newMat[i*3:i*3+3,j*3:j*3+3]
    # finish Hessian by computing diagonal elements
    for i in range(N):
        for j in range(N):
            if j != i:
                newMat[i*3:i*3+3,i*3:i*3+3] -= newMat[j*3:j*3+3,i*3:i*3+3]
    return newMat

# update the Hessian based on the covarDiff updates and alpha 
@jit
def update_hessian(hessian,covarDiff,alpha):
    N = hessian.shape[0]
    for i in range(N-1):
        for j in range(i+1,N):
            hessian[i,j] += alpha * covarDiff[i,j]
            # make sure spring constants stay positive (hessian elements stay negative)
            if hessian[i,j] > 0.0: 
                hessian[i,j] = 0.0
            # symmetrize
            hessian[j,i] = hessian[i,j]
    # update diagonal values
    for i in range(N):
        hessian[i,i] = 0.0
        hessian[i,i] = -np.sum(hessian[i,:])
    return hessian

# compute the difference between predicted and measured particle pair variances
@jit
def compute_henm_diff(newCovar,targetRes,dispVecs):
    # newCovar - the 3Nx3N predicted covariance
    # targetRes - the NxN target variance matrix
    # dispVecs - the NxNx3 displacement vectors
    N3 = newCovar.shape[0]
    N = N3//3
    diffMat = np.zeros((N,N),dtype=np.float64)
    
    for i in range(N-1):
        for j in range(i+1,N):
            temp = newCovar[i*3:i*3+3,i*3:i*3+3] + newCovar[j*3:j*3+3,j*3:j*3+3] - newCovar[i*3:i*3+3,j*3:j*3+3] - newCovar[i*3:i*3+3,j*3:j*3+3].T
            diff = dispVecs[i,j,:]
            diffMat[i,j] = 1.0 / np.dot(diff.T,np.dot(temp,diff)) - targetRes[i,j]
            diffMat[j,i] = diffMat[i,j]
    return diffMat

# Project the pairwise variances computed from the covariance matrix along the pairwise displacement vectors
@jit
def residual_from_covar(covar,dispVecs):
    N3 = covar.shape[0]
    N = N3//3
    residual = np.zeros((N,N),dtype=np.float64)
    for i in range(N-1):
        for j in range(i+1,N):
            temp = covar[i*3:i*3+3,i*3:i*3+3] + covar[j*3:j*3+3,j*3:j*3+3] - covar[i*3:i*3+3,j*3:j*3+3] - covar[i*3:i*3+3,j*3:j*3+3].T
            diff = dispVecs[i,j]
            residual[i,j] = residual[j,i] = 1.0 / np.dot(diff.T,np.dot(temp,diff))
    return residual

# compute pairwise displacement vectors
@jit
def pairwise_disp_vecs(pos):
    N = pos.shape[0]
    dispVecs = np.empty((N,N,3),dtype=np.float64)
    for i in range(N-1):
        for j in range(i+1,N):
            diff = pos[i,:] - pos[j,:]
            diff /= np.linalg.norm(diff)
            dispVecs[i,j,:] = dispVecs[j,i,:] = diff
    return dispVecs
            
# parent routine to perform the hENM procedure
@jit
def perform_henm(targetCovar,avgPos,guessHess = [],alpha=0.1,maxIter=10, thresh=1e-4):
    N3 = targetCovar.shape[0]
    N = N3//3
    # guess a Hessian if none is passed
    if guessHess == []:
        # create initial Hessian
        hessian = -10.0*np.ones((N,N),dtype=np.float64)
        #hessian = reachHessian
        for i in range(N):
            hessian[i,i] = 0.0
            hessian[i,i] = -np.sum(hessian[i,:])
    else:
        hessian = guessHess
    # compute pairwise displacement vectors
    dispVecs = pairwise_disp_vecs(avgPos)
    # convert target covar to target residual
    targetRes = residual_from_covar(targetCovar,dispVecs)
    # iterate
    step = 0
    conv = "FALSE"
    while step < maxIter and conv == "FALSE":
        # project hessian in 3Nx3N
        hessian3N = project_NxN_to_3Nx3N(hessian,dispVecs)
        # invert hessian - using psuedo inverse
        covar3N = np.linalg.pinv(hessian3N,rcond=1e-12)*0.6  # multiply by thermal energy
        # take difference of tensor covars and project into separation vector
        covarDiff = compute_henm_diff(covar3N,targetRes,dispVecs)
        # Check if conveged
        dev = np.linalg.norm(covarDiff)
        if dev < thresh:
            conv = "TRUE"
        else: # update Hessian
            hessian = update_hessian(hessian,covarDiff,alpha)
        step += 1
        print(step, dev)
        sys.stdout.flush()

    return hessian


