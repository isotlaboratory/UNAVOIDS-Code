# This file loads the TXT file outputted by aggregate.py and creates several CSV's saved in [outfolder]. 
# These can be used by our LaTeX code to create latex plots.

import numpy as np
import os

outfolder = "Results_latex"

data_UNAVOIDS = np.ones((4,12,1,6,1,8))*-1 #window axis, norms axis, alpha level axis (not longe used), prevelance axis, transformation axis, method axis
data_Comps = np.ones((4,6,1,4))*-1 #window axis, prevelance axis, transformation axis, method axis

window_sizes = [400,800,1600,3200]
prevelences = ['0.001', '0.002', '0.004', '0.008', '0.016', '0.032']
norms = ["0.0078125", "0.015625", "0.03125", "0.0625", "0.125","0.25","0.5","1","2","4","inf","max"]
transformations = ['none']


for trans_i, trans in enumerate(transformations):

    fp = open("AUCs_"+trans+".txt","r")
    text = fp.read()
    fp.close()

    #split file at each pi header
    Pis =  text.split("____________________________________________________________\n\n############################################################")[1:] 
    for Pi in Pis:

        Pi_comps = Pi.split("UNAVOIDS")[0].split("\n") #grap text containing comparison scores 
        Pi_unavoids = Pi.split("UNAVOIDS\n")[1].split("\np = ") #grap text containing UNAVOIDS scores
        
        #-------------------------------------------------EXTRACT COMPS--------------------------------------------------------------
        pi_i = prevelences.index(Pi_comps[1].split(" ")[-1])
        WS_i = window_sizes.index(int(Pi_comps[4]))

        CompsText = Pi_comps[5:9]

        CompsFloat = np.zeros((1,3)) 
        for n, row in enumerate(CompsText):
            data_Comps[WS_i,pi_i,trans_i,n] = float(row.split(":")[-1])
        
        #-------------------------------------------------EXTRACT UNAVOIDS--------------------------------------------------------------
        
        for norm_i, p in enumerate(Pi_unavoids[:]):
            for method_i, row in enumerate(p.split("\n")[1:9]):
                row = row.split("\t")[1]
                row = np.array([i for i in row.split(" ") if i != ""]).astype(np.float).reshape((1,1))
                data_UNAVOIDS[WS_i,norm_i,:,pi_i,trans_i,method_i] = np.array(row).astype(np.float).reshape(1,1)


for trans_i, trans in enumerate(transformations):
    for WSS_i, WSS in enumerate(window_sizes):
        
        subfolder = str(WSS) + "/"
        
        if not os.path.exists(outfolder+subfolder):
            os.mkdir(outfolder+subfolder)

        for norm_i, norm in enumerate(norms):
            
            #-------------------------------------------------PRINT UNAVOIDS--------------------------------------------------------------
            
            fp = open(outfolder+subfolder+norm+"_"+trans+".csv","w")
            for pi_i in range(6):
                fp.write(str(pi_i+1)+",")
                for method_i in range(7):
                    fp.write(str(data_UNAVOIDS[WSS_i,norm_i,0,pi_i,trans_i,method_i])+",")
                fp.write(str(data_UNAVOIDS[WSS_i,norm_i,0,pi_i,trans_i,7])+"\n")
            fp.close()

            #-------------------------------------------------PRINT COMPS--------------------------------------------------------------
            fp = open(outfolder+subfolder+trans+"_Comps.csv","w")
            for pi_i in range(6):
                fp.write(str(pi_i+1)+",")
                for method_i in range(3):
                    fp.write(str(data_Comps[WSS_i,pi_i,trans_i,method_i])+",")
                fp.write(str(data_Comps[WSS_i,pi_i,trans_i,3])+"\n")
            fp.close()