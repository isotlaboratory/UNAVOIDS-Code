#this file loads and cleans the data from the CICIDS2017 data set

import numpy as np
import pandas as pd
import time
from datetime import datetime

#list of files containing traffic samples
files =["Data/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv",
        "Data/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv",
        "Data/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv",
        "Data/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Data/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Data/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv"]

#used to give numpy the datatypes for each array column, makes opperations faster and soft types columns for error checking
template = ['FID', 'SIP', 0.0, 'DIP', 0.0, 0.0, datetime.strptime("01/01/2000 11:11", '%d/%m/%Y %H:%M')] + [0.0] * 77 + ['label'] #datatypes

#array of samples
X = np.reshape(template, (1,85))

#append samples from each file onto X
n_samps = 0 
for file in files:
    
    try:
        data = pd.read_csv(file, low_memory="false").to_numpy()
    except:
        data = pd.read_csv(file, low_memory="false", encoding="ISO-8859-1").to_numpy()

    n_samps += data.shape[0]
    for i in range(data.shape[0]):
        try:
            data[i,6] = datetime.strptime(data[i,6], '%d/%m/%Y %H:%M')
        except:
            data[i,6] = datetime.strptime(data[i,6], '%d/%m/%Y %H:%M:%S')
    
    #These files are not already sorted by time like the others
    if file in ["Data/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv","Data/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv","Data/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv"]:
        data = data[data[:, 6].argsort()] #sort samples according to column 6, the timestamp

    X = np.append(X, data, axis=0) #append sorted samples

X = np.delete(X, 0, 0) #remove first row, the template

#these files have overlap, so load and sort all at once
files2=["Data/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Data/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]

X2 = np.reshape(template, (1,85))
for file in files2:

    try:
        data = pd.read_csv(file, low_memory="false").to_numpy()
    except:
        data = pd.read_csv(file, low_memory="false", encoding="ISO-8859-1").to_numpy()
        df= pd.read_csv(file, low_memory="false", encoding="ISO-8859-1")

    n_samps += data.shape[0]
    for i in range(data.shape[0]):
        try:
            data[i,6] = datetime.strptime(data[i,6], '%d/%m/%Y %H:%M')
        except:
            data[i,6] = datetime.strptime(data[i,6], '%d/%m/%Y %H:%M:%S')

    X2 = np.append(X2, data, axis=0)


X2 = np.delete(X2, 0, 0) #remove row 0, the template
X2 = X2[X2[:,6].argsort()] #sort Friday afternoon

X = np.append(X, X2, axis=0) #append samples
X= X[X[:,6].argsort()] #double check sort (should already be sorted so this step runs in O(n))


print("These numberse should be the same:")
print("n samps:",n_samps) 
print("X shape[0]:",X.shape[0])

#save uncleaned sorted samples
#np.save("Data/XAll", X)
#exit()

#X = np.load("Data/XAll.npy", allow_pickle=True)


Labels = X[:,-1] #grab labels column

Xid = X[:,[0,1,2,3,5,6,84]]
X = np.delete(X, [0,1,2,3,5,6,84], 1) #remove first 6 columns and label column
X = X.astype(np.float) #convert to floats

#create binary label vector from string labels
Y = []
for i in Labels:
    if i != "BENIGN":
        Y.append(1) 
    else:
        Y.append(0)

delInds = [] #indicies to delete
for i in range(X.shape[0]):
    print(str(i)+"       ", end='\r')
    for j in range(X.shape[1]):
        if (not np.isfinite(X[i,j])) or X[i,j] < -1:  #values less then 0 didn't make sense considering the column type, we set -1 to 0 but samples with values <-1 are removed since they are large negative numbers
            delInds.append(i)
            break
        
X = np.delete(X, delInds, 0)
Xid = np.delete(Xid, delInds, 0)
Y = np.delete(Y, delInds, 0)

for i in range(X.shape[0]):
    print(str(i)+"       ", end='\r')
    for j in range(X.shape[1]):
        if X[i,j] < 0: #to avoid losing too many samples, set -1 values (assumed to indicate an error duirng recording) to 0
            X[i,j] = 0

#save results
np.save("Data/X_IDS", X)
np.save("Data/Y_IDS", Y)
np.save("Data/X_id_IDS", Xid) #saves identifying information for each sample in same order as X

