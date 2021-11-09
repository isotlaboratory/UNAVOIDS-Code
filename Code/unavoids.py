__all__ = ['getAllNCDFs', 'unavoidsScore']

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import multiprocessing as mp
import os
from decimal import Decimal
import warnings
import time
import sys
from sklearn.preprocessing import MinMaxScaler


def getNCDF(X, p, index):
    """ 
    Calculate the NCDF for a single sample using a specified norm 

    Parameters:
    X (n x m numpy array): an array containing n samples and m feature values, assumed to be normalized between 0 and 1
    p (float): the norm to use when calculating the distance between points
    index (int): the index of the sample in X which we are finding the NCDF for

    Returns:
    NCDFxi (1 x n numpy array): where the n^th value equals NCDF_xi(n)
    """

    n = X.shape[0]
    d = X.shape[1]
    
    with warnings.catch_warnings():
        try:
            warnings.filterwarnings('error')
            
            NCDFxi = np.zeros((1, n))  # matrix to hold NCDF
            
            if p == np.inf:
                NCDFxi[0,:] = ( np.max(np.abs(X[index,:]-X[:,:]), axis=1)) #calculate Chebyshev distance between sample X[i] and X[j != i]
            else:
                NCDFxi[0,:] = ( np.sum(np.abs(X[index,:]-X[:,:])**p, axis=1)**(1.0/float(p))) #calculate p-norm of samples X[i] and X[j != i]
        
            #normalize by max volumne
            maxNorm = np.max(NCDFxi[0,:])
            if maxNorm  > 0: 
                NCDFxi[0,:] = NCDFxi[0,:]/maxNorm     
        
            NCDFxi = np.sort(NCDFxi, axis=1)

        except Warning as e:
            print("Warning: "+str(e)+" -> switching to from numpy to Decimal library implementation, expect speed decrease.\n\tAny further warnings mean results may be incorrect.")
            
            NCDFxi = []  # array to hold NCDF
            
            for I in range(n):
                if p == np.inf:
                    NCDFxi.append(( np.max(np.abs(X[index,:]-X[I,:])) )) #calculate Chebyshev distance between sample X[i] and X[j != i]
                else:
                    NCDFxi.append(Decimal(( np.sum(np.abs(X[index,:]-X[I,:])**p)))**Decimal(1.0/float(p))) #/ (d**(1.0/p)) #calculate p-norm of samples X[i] and X[j != i] 

            #normalize by max volume
            maxNorm = max(NCDFxi)
            if maxNorm > 0:
                for I in range(n):
                    NCDFxi[I] = NCDFxi[I]/maxNorm    
            
            NCDFxi = np.array(NCDFxi.sort()).reshape((1, n))
    
    return NCDFxi

def getAllNCDFs(X, p=0.0625, ncpus=4):
    """ 
    Calculate the NCDF for all samples in parallel using a specified norm 

    Parameters:
    X (n x m numpy array): an array containing n samples and m feature values
    p (float): the norm to use when calculating the distance between points
    ncpus (int): the number of parallel processes

    Returns:
    NCDF (n x n numpy array): the n^th row equals the NCDF for the nth sample in X
    """

    if np.amax(X) != 1.0 or np.amax(X) != 0:

        #normalize features between 0 and 1
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
                        
    #catch overflows, underflows and invalid values and invalid division 
    np.errstate(all='raise')
    np.seterr(all='raise')
    
    #get NCDFs
    pool = mp.Pool(processes=ncpus)
    func = partial(getNCDF, X, p) #pass X and p as first two args of getNCDF
    result = pool.map(func, range(X.shape[0])) #run getNCDF in parallel across each sample
    pool.close()
    pool.join()

    return np.reshape(result, (X.shape[0],X.shape[0]))

def getBetaFractions(NCDFs_L, BetaSorted, BetaRanks, fraction_WSS, index):
    """ 
    Calculate the UNVAOIDS outlier score for a given sample using the fractions of all gaps method

    Parameters:
    NCDFs_L (n x L numpy array): an array containing the intercepts for n NCDFs at L beta levels
    BetaSorted (n x L numpy array): the same as NCDFs_L but the intercepts are sorted along the L beta levels (columnwise sort of NCDFs_L) 
    BetaRanks (n x L numpy array): the same as NCDFs_L but the value at (i,j) is replaced with the rank of NCDFs_L[i,j] on a given beta horizontal
    Fractions_WSS: the number of nearest intercepts to be encompassed by the gap whose size will be the score for the given beta and NCDF's intercept
    index (int): the index of the NCDF in X which we are finding the score for

    Returns:
    score (1 x 1 numpy array): equal to the score for current samples across all beta levels
    """

    n = NCDFs_L.shape[0] #number of Betas
    L = NCDFs_L.shape[1] #number of NCDFs

    k_gaps = np.zeros((L,1))

    #for each column
    for col in range(L):
        obser_intercept = NCDFs_L[index,col] #get intercept of this NCDF
        obser_rank = BetaRanks[index,col] #get rank of intercept for this NCDF

        #get nearest(by rank) fraction_WSS * 2 intercepts
        if obser_rank - fraction_WSS < 0:
            bottom = 0
            top =  obser_rank + fraction_WSS + 1 - (obser_rank - fraction_WSS)
        elif obser_rank + fraction_WSS + 1 > n:
            bottom = obser_rank - fraction_WSS - ((obser_rank + fraction_WSS + 1) - n)
            top =  n
        else:
            bottom = obser_rank - fraction_WSS 
            top = obser_rank + fraction_WSS + 1

        #sort only the gaps to the k_max * 2 nearest intercepts 
        gaps = np.sort(np.abs(BetaSorted[bottom:top,col] - obser_intercept))

        #get gaps for each Fraction
        k_gaps[col,0] = gaps[fraction_WSS]

    #get largest gap matrix
    score = np.amax(k_gaps,axis=0)
    
    return score


def getBetasHist(NCDFs_L, BetaSorted, index):

    """ 
    Calculate the UNVAOIDS outlier score for a given sample using the histogram method

    Parameters:
    NCDFs_L (n x L numpy array): an array containing the intercepts for n NCDFs at L beta levels
    BetaSorted (n x L numpy array): the same as NCDFs_L but the intercepts are sorted along the L beta levels (columnwise sort of NCDFs_L) 
    index (int): the index of the NCDF in X which we are finding the score for

    Returns:
    score (1 x 1 numpy array): equal to the score for current samples across all beta levels
    """

    n = NCDFs_L.shape[0] #number of NCDFs
    L = NCDFs_L.shape[1] #number of Betas

    beta_max = 0 #the highest score of the beta levels

    n_bins = n * 0.05
    step = 1/n_bins
    edges_s = np.arange(0,1.01,step)

    if type(NCDFs_L[0,0]) == Decimal:
        step = Decimal(step)
        edges_s = np.array([Decimal(i) for i in edges_s])
    
    for col in range(1, L-1):
        obser_intercept = NCDFs_L[index,col] #intercept of observation
        hrzntl = NCDFs_L[:,col]              #current beta level

        #center observation intercept in bin with width 0.05
        if obser_intercept < step: #avoid underflow errors
            n_le = 0
        else:
            n_le = int(obser_intercept/step)

        edges = edges_s + ((obser_intercept - edges_s[n_le]) - (step/2))

        if edges[-1] > 1:
           edges = np.append([0.0], edges)
           edges[-1] = 1.01
           n_le += 1
        elif edges[-1] < 1:
           edges = np.append(edges, [1.01])
           edges[0] = 0.0

        observed_bins = np.where(BetaSorted[:,col] > edges[1], np.minimum(((BetaSorted[:,col]-edges[1])/step).astype('int') + 1, len(edges)-1), 0)
        u, c = np.unique(observed_bins, return_counts=True)
        hist = np.zeros((len(edges)-1,))
        hist[u] = c

        beta = np.sum(np.where(hist > hist[n_le], hist, 0))/n

        #compare with best score so far
        if beta > beta_max:
            beta_max = beta

    return np.array(beta_max).reshape((1,1))


def unavoidsScore(X, precomputed=False, p=0.0625, returnNCDFs=True, method="fractions", r=0.01,  L=100, ncpus=4):

    """ 
    Calculate the UNVAOIDS outlier score for a all samples

    Parameters:
    X (n x m numpy array): an array containing n samples and m feature values
    precomputed (boolena): if True, X is assumed to be the NCDF array returned by getAllNCDFs
    p (float): the norm to use when calculating the distance between points
    returnNCDFs(boolean): if True, NCDF array is returned along with outlier scores
    method (string): specifies which method to use for calculating outlier scores; either "fractions" or "histogram"
    r: percentage of nearest intercepts to be encompassed by the gap whose size will be the score for the given beta and NCDF's intercept when using the "fractions" method
    L (int): the number of beta levels to use
    ncpus (int): the number of parallel processes

    Returns:
    scores (n x 1 numpy array): where the n^th element is equal to the score for the n^th sample in X
    NCDFs (n x n numpy array): only returned if returnNCDFs == True. The n^th row equals the NCDF for the nth sample in X
    """

    if precomputed == False:
        NCDFs = getAllNCDFs(X, p)
    else:
        NCDFs = X

    WSS = NCDFs.shape[0]

    Lindexes = np.unique(np.append(np.floor(np.arange(0,L)*(WSS/L)), WSS-1).astype(int)) #indicies of beta levels 
    Fractions_WSS = int((r * WSS))  #convert percentage to portion of window size

    NCDFs_L = NCDFs[:, Lindexes] #for current norm, grab all NCDF intercepts with all L beta levels 
    BetaSorted = np.sort(NCDFs_L, axis=0) #sort intercepts along beta level
    BetaRanks = np.argsort((np.argsort(NCDFs_L, axis=0)), axis=0) #get ranks of intercepts
    
    if method == "fractions":
        #get score for each sample and all fractions using Fractions method
        pool = mp.Pool(processes=ncpus)
        func = partial(getBetaFractions, NCDFs_L, BetaSorted, BetaRanks, Fractions_WSS)
        scores = np.array(pool.map(func, range(NCDFs_L.shape[0])))
        pool.close()
        pool.join()
    
    elif method == "histogram":
        #get best beta using Histogram approaches
        pool = mp.Pool(processes=ncpus)
        func = partial(getBetasHist, NCDFs_L, BetaSorted)
        scores = np.array(pool.map(func, range(0, NCDFs_L.shape[0])))
        pool.close()
        pool.join()

    if returnNCDFs == False:
        return scores.reshape((NCDFs.shape[0], -1))
    else:
        return scores.reshape((NCDFs.shape[0], -1)), NCDFs



