import numpy as np
import matplotlib.pyplot as plt
import sys
import unavoids
from joblib import load



if __name__ == "__main__":

    X = load("simData.joblib")[:,:2]
    Y = np.zeros((X.shape[0],))
    Y[-3:] = 1 #last thre sampls are outliers

    betas, NCDFs = unavoids.unavoidsScore(X, precomputed=False, returnNCDFs=True)

    print(betas[-5:])


    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(NCDFs.shape[0]):
        NCDFxi = np.array((NCDFs[i,:]), dtype=float)
        
        #if i == 564:
        #    ax.step(NCDFxi,np.arange(565)/565, lw=0.25, color="red",  alpha=1) #outlier 
        #elif i == 563:
        #    ax.step(NCDFxi,np.arange(565)/565, lw=0.25, color="green",  alpha=1) #outlier
        #elif i == 562:
        #    ax.step(NCDFxi,np.arange(565)/565, lw=0.25, color="blue",  alpha=1) #outlier
        if betas[i] >= 0.01740045:
            ax.step(NCDFxi,np.arange(565)/565, lw=1, color="red",  alpha=1) #outlier 
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