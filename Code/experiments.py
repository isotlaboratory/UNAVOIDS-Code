# This script runs UNAVIODS and some comparison algorithms across a grid of parameters on a window specified by a command line argument.
#
# Expects n x m feature array stored in [Datafld]X.npy and an n x 1 array of binary labels in {0,1} stroed in [Datafld]Y.npy
#   - also reqires the folder [Pisfld] to contain Pis_[i].npy for each i, where i corresponds to the index of the i^th prevelance in the array Pis
#   - each Pis_[i].npy contains a n' < n length vector of indicies, such that the prevelance of outliers in the subset pointed to by these indicies corresponds to the ith
#     value of Pis.
#
# UNAVOIDS results are saved in ROCAUCfolder and comparison results are saved in CompFolder

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
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
#______________________________________________________FUNCTIONS______________________________________________________

def getNCDF(X, p, index):

    n = X.shape[0]
    d = X.shape[1]
    
    with warnings.catch_warnings():
        try:
            warnings.filterwarnings('error')
            
            NCDFxi = np.zeros((1, n))  # matrix to hold NCDF
            
            if p == np.inf:
                NCDFxi[0,:] = ( np.max(np.abs(X[index,:]-X[:,:]), axis=1)) #calculate Chebyshev distance between sample X[i] and X[j != i] normalized by the max volume 
            else:
                NCDFxi[0,:] = ( np.sum(np.abs(X[index,:]-X[:,:])**p, axis=1)**(1.0/float(p))) #/ (d**(1.0/p)) #calculate p-norm of samples X[i] and X[j != i] normalized by the max volume 
        
            maxNorm = np.max(NCDFxi[0,:])
            if maxNorm  > 0:
                NCDFxi[0,:] = NCDFxi[0,:]/maxNorm     
        
            NCDFxi = np.sort(NCDFxi, axis=1)

        except Warning as e:
            print("Warning: "+str(e)+" -> switching to from numpy to Decimal library implementation, expect speed decrease.\n\tAny further warnings mean results may be incorrect.")
            
            NCDFxi = []  # array to hold NCDF
            
            for I in range(n):
                if p == np.inf:
                    NCDFxi.append(( np.max(np.abs(X[index,:]-X[I,:])) )) #calculate Chebyshev distance between sample X[i] and X[j != i] normalized by the max volume 
                else:
                    NCDFxi.append(Decimal(( np.sum(np.abs(X[index,:]-X[I,:])**p)))**Decimal(1.0/float(p))) #/ (d**(1.0/p)) #calculate p-norm of samples X[i] and X[j != i] normalized by the max volume 

            maxNorm = max(NCDFxi)
            if maxNorm > 0:
                for I in range(n):
                    NCDFxi[I] = NCDFxi[I]/maxNorm    
            
            NCDFxi.sort()
    
    return NCDFxi

#______________________________________________________FUNCTIONS______________________________________________________

def getBetaFractions(NCDFs_L, BetaSorted, BetaRanks, Fractions_WSS, obser):
    #distance which encompasses some fraction of all intercepts approach

    n = NCDFs_L.shape[0] #number of Betas
    L = NCDFs_L.shape[1] #number of NCDFs

    k_max = Fractions_WSS[-1]  #get largest Fraction

    k_gaps = np.ones((L,len(Fractions_WSS))) ** -1

    #for each column
    for col in range(L):
        obser_intercept = NCDFs_L[obser,col] #get intercept of this NCDF
        obser_rank = BetaRanks[obser,col] #get rank of inercept for this NCDF
        
        #get nearest(by rank) k_max * 2 intercepts
        if obser_rank - k_max < 0:
            bottom = 0
            top =  obser_rank + k_max + 1 - (obser_rank - k_max)
        elif obser_rank + k_max + 1 > n:
            bottom = obser_rank - k_max - ((obser_rank + k_max + 1) - n)
            top =  n
        else:
            bottom = obser_rank - k_max 
            top = obser_rank + k_max + 1

        #sort only the gaps to the k_max * 2 nearest intercepts 
        gaps = np.sort(np.abs(BetaSorted[bottom:top,col] - obser_intercept))

        #get gaps for each Fraction
        k_gaps[col,:] = gaps[Fractions_WSS]
        
    #get index of beta with largest gap metric
    #beta_ks = np.argmax(k_gaps,axis=0)
    #beta_avg = np.argmax(np.sum(k_gaps, axis=1),  axis=0)
    #betas = np.append(beta_ks, beta_avg) 
    
    #get largest gap metrix
    beta_ks = np.amax(k_gaps,axis=0)
    beta_avg = np.amax(np.sum(k_gaps, axis=1),  axis=0)
    betas = np.append(beta_ks, beta_avg) 

    return betas

def getBetas_AllAvg(NCDFs_L, beta):

    n = NCDFs_L.shape[0]

    hrzntl = NCDFs_L[:,beta]
    gaps = np.abs(hrzntl.transpose() - np.reshape(hrzntl, (hrzntl.shape[0],1)))
    #np.fill_diagonal(gaps, np.inf)
    
    gaps = np.sum(gaps, axis=1)

    return gaps

def getBetasHist(NCDFs_L, BetaSorted, obser):

    n = NCDFs_L.shape[0] #number of NCDFs
    L = NCDFs_L.shape[1] #number of Betas

    beta_max1 = 0 #the highest score of the beta levels
    beta_max2 = 0 #the highest score of the beta levels

    n_bins = n * 0.05
    step = 1/n_bins
    
    for col in range(1, L-1):
        obser_intercept = NCDFs_L[obser,col] #intercept of observation
        hrzntl = NCDFs_L[:,col]              #current beta level

        #center obeservation intercept in bin with width 0.05
        lb = obser_intercept - 0.025       
        ub = obser_intercept + 0.025

        #create the rest of the bins between 0 and 1 with widths 0.05, edge bins may be cut off by subceeding 0 or exceeding 1
        edges = [0,1.01]
        n_le = 1 #number of edges below observation
        while True:
            if lb <= 0:
                break 
            else:
                n_le += 1
                edges.append(lb)
                lb -= step

        while True:
            if ub >= 1:
                break 
            else:
                edges.append(ub)
                ub += step
        edges.sort() #sort edges

        #create histogram and get bin counts
        hist = np.zeros((len(edges)-1,1))
        cur_count = 0
        cur_edge = 1
        
        for intercept in BetaSorted[:,col]:
            while True:
                if intercept <= edges[cur_edge]: #if current sample is less then cur edge, add to cur bin
                    hist[cur_edge-1] += 1 #increment bin counter
                    break #break when you find the sample's bin
                else: #else look at next edege/bin
                    cur_edge +=1

        #determine score
        score = 0 
        for i in hist:
            if (hist[n_le-1] < i[0]):
               score += i[0]
        beta1 = score/n

        #prob, edges = np.histogram(hist, bins=5, density=True)
        counts, edges = np.histogram(hist, bins=5)
        prob = counts/hist.shape[0]
        edges[-1] = edges[-1] + 1
        beta2 = 1 - prob[np.digitize(hist[n_le-1], bins=edges)-1][0]
        
        #compare with best score so far
        if beta1 > beta_max1:
            beta_max1 = beta1

        if beta2 > beta_max2:
            beta_max2 = beta2

    return np.array([beta_max1, beta_max2]).reshape((2,1))

#_________________________________________________________MAIN____________________________________________________________


window = int(sys.argv[1]) #index of the window of samples used in the current job

debug = False #True to print debug information
plot = False #if Debug is also True, will plot NCDFs with outliers in red

ncpus = 4 #number of parallel processes to use

L = 100 #number of beta levels to compute gap at @@@

CompFolder = "Comps/" #where to save comparison algorotihm AUC scores
ROCAUCfolder = "ROCAUCs" #where to save Unavoids AUC scores
Pisfld = "PiIndexes/" #contains several index arrays which give a specific prevelance over the entire dataset, ensures windows contain the same samples across successive runs  
Datafld  = "Data/" #where the label and feature matricies are stored for the entire dataset

WindowSampleSizes = [400, 800, 1600, 3200]
Fratctions = [0.01, 0.02, 0.04, 0.08]
Norms = [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, np.inf] 
Logs = [False, True]
Pis = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 556541/2271054]

if __name__ == '__main__':   

    for WSS in WindowSampleSizes:
 
        subfolder = str(WSS)+"/" #subfolder to save results for current window size

        Lindexes = np.unique(np.append(np.floor(np.arange(0,L)*(WSS/L)), WSS-1).astype(int)) #indicies of beta levels 

        Fractions_WSS = (np.array(Fratctions) * WSS).astype(int) #convert percentages to proptions of window size

        for n, pi in enumerate(Pis):

            pi_inds = np.load(Pisfld+"Pis_"+str(n)+".npy") #load index array for current prevelance = pi

            if WSS*(window+1) >= pi_inds.shape[0]: #if window out of bounds
                continue #try next prevelance
            
            pi_inds_WSS = pi_inds[WSS*window:WSS*(window+1)]  #grab list of indicies for current window

            Y = np.load(Datafld+"Y.npy", allow_pickle=True)[pi_inds_WSS] #grab window's labels
            X = np.load(Datafld+"X.npy", allow_pickle=True)[pi_inds_WSS] #grab window's samples
            
            if np.sum(Y) != 0 and np.sum(Y) != WSS: #skip windows with no outliers, or all outliers

                for log in Logs: #no transform must come first becuase X is not reloaded between loops

                    #apply transformation
                    if log:
                        desc = str(window)+"_"+str(n) + "_Log_"
                        X = np.log(X + np.finfo(float).eps)
                    else:
                        desc = str(window)+"_"+str(n) + "_none_"
                    
                    #normalize features between 0 and 1
                    scaler = MinMaxScaler()
                    X = scaler.fit_transform(X)

                    #___________________Calculate AUC for Comparison Algorithms_________________________________________________________#
                    
                    if not os.path.exists(CompFolder+subfolder):
                        os.mkdir(CompFolder+subfolder) 

                    fp = open(CompFolder+subfolder+desc+"COMPs","w+") #open file to write scores to

                    LOF1 = np.zeros((1,X.shape[0])) #save scores for k 10,11...36 accroding to method published in original paper
                    LOF2 = np.zeros((1,X.shape[0])) #save scores for k 10,15,..80
                    for k in range(10,36):
                        clf = LocalOutlierFactor(n_neighbors=k)
                        clf.fit(X)
                        LOFscores = -clf.negative_outlier_factor_
                        LOF1 = np.maximum(LOF1, LOFscores)
                        if k % 5 == 0:
                            LOF2 = np.maximum(LOF2, LOFscores)
                    for k in range(40,81,5):
                        clf = LocalOutlierFactor(n_neighbors=k)
                        clf.fit(X)
                        LOFscores = -clf.negative_outlier_factor_
                        LOF2 = np.maximum(LOF2, LOFscores)

                    #calculate and write results for LOF1 and LOF2
                    AUC = roc_auc_score(Y, LOF1[0])
                    fp.write("LOF1: "+str(AUC)+"\n")
                    AUC = roc_auc_score(Y, LOF2[0])
                    fp.write("LOF2: "+str(AUC)+"\n")
                    
                    #calculate and write results for ABOD
                    clf = ABOD(n_neighbors=160)
                    clf.fit(X)
                    ABODscores = clf.decision_scores_
                    np.nan_to_num(ABODscores, copy=False, nan = np.nanmin(ABODscores))
                    AUC = roc_auc_score(Y, ABODscores)
                    fp.write("FastABOD: "+str(AUC)+"\n")

                    #calculate and write results for IForest
                    clf = IForest(n_estimators=100)
                    clf.fit(X)
                    IFscores = clf.decision_scores_
                    AUC = roc_auc_score(Y, IFscores)
                    fp.write("Iso_For: "+str(AUC)+"\n")
                    fp.close()
                    

                    #_______________________________NCDFs___________________________________________________________________________#

                    if debug:
                        start_time = time.time()

                    NCDFs_All = np.zeros((len(Norms),X.shape[0],X.shape[0])) #array of NCDFs for each norm 
                    for p_i, p in enumerate(Norms): 
                        
                        if debug:
                            ncdf_time = time.time()
                        
                        #catch overflows, underflows and invalid values and invalid division 
                        np.errstate(all='raise')
                        np.seterr(all='raise')
                        
                        #get NCDFs
                        pool = mp.Pool(processes=ncpus)
                        func = partial(getNCDF, X, p) #pass X and p as first two args of getNCDF
                        result = pool.map(func, range(X.shape[0])) #run getNCDF in parallel across each sample
                        pool.close()
                        pool.join()

                        NCDFs_All[p_i] = np.reshape(result, (X.shape[0],X.shape[0]))
                        del result
                      
                        if debug:
                            NCDFs = NCDFs_All[p_i]

                            print("NCDF time:", time.time() - ncdf_time)

                            if plot == True: 
                                fig = plt.figure()
                                ax = fig.add_subplot(111)

                                for i in range(NCDFs.shape[0]):
                                    NCDFxi = np.array((NCDFs[i,:]), dtype=float)
                                    
                                    if Y[i] == 1:
                                        ax.step(NCDFxi,np.arange(WSS)/WSS, lw=0.1, color="red",  alpha=1) #outlier
                                    else:
                                        ax.step(NCDFxi,np.arange(WSS)/WSS, lw=0.1, color="grey",  alpha=1) #regular datum #outlier [0.5, 0.5]
                                    
                                    xleft, xright = ax.get_xlim()
                                    ybottom, ytop = ax.get_ylim()
                                    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop)))

                                plt.show()
                                plt.close()

                    #_______________________________UNAVOIDS Outlier Scores_________________________________________________________#
                    
                    all_betas = np.zeros((len(Norms), NCDFs_All.shape[1] ,8), dtype=float) #array of outlier scores for each norm, sample, and outlier detection method
                    for p_i, p in enumerate(Norms):

                        NCDFs_L = NCDFs_All[p_i][:, Lindexes] #for current norm, grab all NCDF intercepts with all L beta levels 
                        BetaSorted = np.sort(NCDFs_L, axis=0) #sort intercepts along beta level
                        BetaRanks = np.argsort((np.argsort(NCDFs_L, axis=0)), axis=0) #get ranks of intercepts

                        if debug:
                            frac_time = time.time()
                        
                        #get score for each sample and all fractions using Fractions method
                        pool = mp.Pool(processes=ncpus)
                        func = partial(getBetaFractions, NCDFs_L, BetaSorted, BetaRanks, Fractions_WSS)
                        cur_betas = np.array(pool.map(func, range(NCDFs_L.shape[0])))
                        pool.close()
                        pool.join()
                        
                        if debug:
                            print("Fractions time:", time.time() - frac_time)
                            allgaps_time = time.time()

                        #get scores using Average of All Gaps approach
                        pool = mp.Pool(processes=ncpus)
                        func = partial(getBetas_AllAvg, NCDFs_L)
                        result = np.array(pool.map(func, range(NCDFs_L.shape[1])))
                        pool.close()
                        pool.join()
                        cur_betas = np.append(cur_betas, np.reshape( np.max(result, axis=0), (NCDFs_L.shape[0], 1)), axis=1) #use max score across all betas
                        del result

                        if debug:
                            print("All averaged time:",time.time() - allgaps_time)
                            hist_time = time.time()
                        
                        #get best beta using Histogram approaches
                        pool = mp.Pool(processes=ncpus)
                        func = partial(getBetasHist, NCDFs_L, BetaSorted)
                        result = np.array(pool.map(func, range(0, NCDFs_L.shape[0])))
                        cur_betas = np.append(cur_betas, np.reshape(result, (NCDFs_L.shape[0], 2)), axis=1)
                        pool.close()
                        pool.join()
                        del result
                        
                        if debug:
                            print("time Histogram, all iters:",time.time() - hist_time)

                        all_betas[p_i] = cur_betas
                        del cur_betas

                    if debug:
                        print("UNVAOIDS total time:", time.time() - start_time)

                    #_______________________________Calculate AUC for UNAVOIDS______________________________________________________#
                    if not os.path.exists(ROCAUCfolder+subfolder):
                        os.mkdir(ROCAUCfolder+subfolder)

                    fpa = open(ROCAUCfolder+subfolder+desc+"ROCAUCs.txt","w") #open file to write scores to
                    for p_i, p in enumerate(Norms): 
                        fpa.write("P_ind:"+str(p_i)+"\n")
                        for method in range(8):
                            fpa.write(str(method)+",")

                            AUC = roc_auc_score(Y, all_betas[p_i,:,method]) #write AUC for current norm and method
                            fpa.write(str(AUC)+"\n")
                        fpa.write("\n")
                    

                    fpa.write("P_ind:"+"max"+"\n") 
                    max_betas = np.amax(all_betas, axis=0)
                    for method in range(8):
                        fpa.write(str(method)+",")

                        AUC = roc_auc_score(Y, max_betas[:,method]) #write AUC using max score across all norms
                        fpa.write(str(AUC)+"\n")

                    fpa.write("\n")
                    fpa.close()

                    del NCDFs_All
                    del all_betas

            del X
            del Y