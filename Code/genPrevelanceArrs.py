
# Thus script creates Pis_[i].npy for each i, where i corresponds to the index of the i^th prevelance in the array Pis
# each Pis_[i].npy contains a n' < n length vector of indicies, such that the prevelance of outliers in the subset pointed to by these indicies corresponds to the ith
# value of Pis.

import numpy as np
from scipy import stats 

Y = np.load("Data/Yml_IDS.npy", allow_pickle=True)

Pisfld = "PiIndexes/"

Y_0_inds = np.where(Y == 0)[0]
Y_1_inds = np.where(Y == 1)[0]

n_0 = Y_0_inds.shape[0]

Pis = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.19682486353]

print(Y.shape)

for i, pi in enumerate(Pis):
    

    n_1 = round((pi/(1-pi))*n_0) #number of malicious for desired prevalence

    Y_1_inds_samp = np.reshape(np.random.choice(Y_1_inds, size=n_1, replace=False), (n_1,)) #sample malicious
    pi_inds = np.random.shuffle(np.append(Y_0_inds, Y_1_inds_samp)) #append and shuffle
    
    np.save(Pisfld+"Pis_"+str(i), pi_inds)

    print(i, round(Y_1_inds_samp.shape[0]/(pi_inds.shape[0]), 5) )