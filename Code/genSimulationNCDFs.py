#this script finds the NCDFs for simulated data using different norms and numbers of features, then creates CSV files for LaTeX to plot the NCDFs

import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from joblib import load
from functools import partial
import multiprocessing as mp
import warnings

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

#_________________________________________________________MAIN____________________________________________________________

CompAlgorithms = False
debug = True
ncpus = 4
L = 100 #number of beta levels (I think)

NCDFfolder = "SimNCDFs/"
PDFfolder = "neg5pdfNCDFsSVG/" #"pdfNCDFsSVG/"

Features =[2, 4, 8, 16, 32, 64, 128, 256, 512]
Norms = [0.0078125, 0.0078125,0.0625,0.125,0.25,0.5, 1, 2, 4, np.inf]


if __name__ == '__main__':

    X_all = load("simData.joblib")
    Y = np.zeros((X_all.shape[0],))
    Y[-3:] = 1 #last thre sampls are outliers

    for n_feats in Features:

        desc1 = str(n_feats)+"_"

        X = X_all[:,:n_feats] #grab first n_feats features
            
        for p_i, p in enumerate(Norms):
            
            desc2 = str(p_i)+"_"
            
            #catch overflows, underflows and invalid values and invalid division 
            np.errstate(all='raise')
            np.seterr(all='raise')
            
            #get NCDFs
            pool = mp.Pool(processes=ncpus)
            func = partial(getNCDF, X, p) #pass X and p as first two args of getNCDF
            result = pool.map(func, range(X.shape[0])) #run getNCDF in parallel across each sample
            pool.close()
            pool.join()

            NCDFs = np.reshape(result, (X.shape[0],X.shape[0]))
            
            #np.save(NCDFfolder+desc1+desc2+"NCDFs.npy",NCDFs)
                
            fig = plt.figure()
            ax = fig.add_subplot(111)

            for i in range(NCDFs.shape[0]):
                NCDFxi = np.array((NCDFs[i,:]), dtype=float)
                
                if i == 564:
                    ax.step(NCDFxi,np.arange(565)/565, lw=0.25, color="red",  alpha=1) #outlier 
                elif i == 563:
                    ax.step(NCDFxi,np.arange(565)/565, lw=0.25, color="green",  alpha=1) #outlier
                elif i == 562:
                    ax.step(NCDFxi,np.arange(565)/565, lw=0.25, color="blue",  alpha=1) #outlier
                else:
                    ax.step(NCDFxi,np.arange(565)/565, lw=0.05, color="grey",  alpha=1) #regular datum 
                
                xleft, xright = ax.get_xlim()
                ybottom, ytop = ax.get_ylim()
                ax.set_aspect(abs((xright-xleft)/(ybottom-ytop)))
                
            plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            #plt.savefig(PDFfolder+desc1+desc2+"NCDFs.svg", format='svg', bbox_inches = 'tight', pad_inches = 0)
            plt.show()
            plt.close()
    del X
    del Y