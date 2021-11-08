#this file loads and cleans the data from the CIRA-CIC-DoHBrw-2020 data set

import numpy as np
import pandas as pd
import time
from datetime import datetime


X_benign = pd.read_csv("Data/Total-CSVs/l1-nondoh.csv").to_numpy() #load nonDoH files

X_malicious = pd.read_csv("Data/Total-CSVs/l2-malicious.csv").to_numpy() #load malicious DoH

#append malicious and non malicious samples
X = np.append(X_benign, X_malicious, axis=0)
Xid = X[:,:5] 
X = X[:,5:-1] #remove first four columns and last column
Y = np.append(np.zeros((X_benign.shape[0],1)), np.ones((X_malicious.shape[0], 1)), axis=0) #create label vector

#convert timestamps to datetime objects
for x in Xid:
    x[-1] = datetime.strptime(x[-1], "%Y-%m-%d %H:%M:%S")

#sort by time stamp
sortIndeces = np.argsort(Xid[:,-1])
X = X[sortIndeces]
Y = Y[sortIndeces]
Xid = Xid[sortIndeces]

#remove rows with nonfininte numbers
removeInd = []
for n, x in enumerate(X):
    for m, i in enumerate(x):
        if not np.isfinite(i):
            if n not in removeInd:
                removeInd.append(n)

X = np.delete(X, removeInd, 0)
Y = np.delete(Y, removeInd, 0)
Xid = np.delete(Xid, removeInd, 0)

np.save("Data/X_DoH", X)
np.save("Data/Y_DoH", Y)
np.save("Data/X_id_DoH", Xid) #saves identifying information for each sample in same order as X