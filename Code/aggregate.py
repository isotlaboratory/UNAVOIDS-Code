#This script aggregates the scores outputted by expriments.py

import os
import numpy as np
import sys

ROCAUCfolder = "ROCAUCs/" #Unavoids AUC scores
COMPfolder = "Comps/" #Comparison algorithm AUC scores

transformation = "none" #indicates wether to aggregate transformation scores or not

comps = ["LOF1:   ","LOF2:   ","FastABOD:","Iso_For:"]
methods = ["r=0.01: \t", "r=0.02: \t", "r=0.04: \t", "r=0.08: \t", "r avg:  \t", "all avg:\t", "hist:   \t", "hist 2 :\t"]
ps = ["0.0078125","0.015625", "0.03125", "0.0625","0.125","0.25","0.5", "1", "2","4","np.inf","max"]
Pis = ["0.001", "0.002", "0.004", "0.008", "0.016", "0.032", "0.064", "0.128","0.256","full"]

subdirs = sorted(os.listdir(ROCAUCfolder), key=lambda elem: int(elem)) #list subdirs in order of window size

fpe = open("ErrorAggregate.txt","w+") #output errors
fpo = open("AUCs_"+transformation+".txt", "w+")
for sub_n, subfolder in enumerate(subdirs): #for each Window size

    K = np.zeros((6,)) #keep track of number of files processed for each pi
    AUCs = np.zeros((6,12,8,1)) #pi axis, norm axis, method axis, score axis
    COMPs = np.zeros((6,4,1)) #pi axis, method axis, score axis

    progress=0

    for UNAVOIDSresult in os.listdir(ROCAUCfolder+subfolder): #for each results file

        Pi_ind = int(UNAVOIDSresult.split("_")[1]) #get pi param index

        if (Pi_ind > 5): #do not use pi larger then 0.32
            continue

        if UNAVOIDSresult.split("_")[2] != transformation: #use file if it matches transformation setting, otherwise skip
            continue

        fpU = open(ROCAUCfolder+subfolder+"/"+UNAVOIDSresult, "r") #open file 
        unavoids_rows = fpU.read().split("\n") #read rows as list of rows
        fpU.close()

        #check file integrity
        if len(unavoids_rows) != 122:
            fpe.write("Error: "+ROCAUCfolder+subfolder+"/"+UNAVOIDSresult+":\tincorrect number of rows in UNAVOIDS scores\n")
            continue
        cont = False
        for row_n, row in enumerate(unavoids_rows):
            if row_n in [0,9,10,19,20,29,30,39,40,49,50,59,60,69,70,79,80,89,90,99,100,109,110,119,120,121]:
                continue
            if len(row.split(",")) != 2:
                cont = True
        if cont == True:
            fpe.write("Error: "+ROCAUCfolder+subfolder+"/"+UNAVOIDSresult+":\tincorrect number of columns in UNAVOIDS scores\n")
            continue
        
        #Get LOF, ABOD, and Isolation Forest scores for corresponding pi and window size
        COMPresult = UNAVOIDSresult.split("_R")[0] + "_COMPs" #get comp results file name
        try:
           fpC = open(COMPfolder+subfolder+"/"+COMPresult, "r") #open LOF file
        except:
           fpe.write("Error: "+COMPfolder+subfolder+"/"+COMPresult+":\tfile not found"+"\n")
           continue
        comps_rows = fpC.read().split("\n") #read rows as list of rows
        fpC.close()

        #check file integrity
        if len(comps_rows) != 5:
            fpe.write("Error:"+ROCAUCfolder+subfolder+"/"+COMPresult+":\tincorrect number of rows in comparison scores\n")
            print("not 5 rows")
            continue
        cont = False
        for n_row, row in enumerate(comps_rows[:-1]):
            if len(row.split(": ")) != 2:
                cont = True
        if cont == True:
            fpe.write("Error:"+ROCAUCfolder+subfolder+"/"+COMPresult+":\tincorrect number of columns in comparison scores\n")
            continue

        K[Pi_ind]+=1 #increment counter for current value of pi

        curCOMPs = np.zeros((4,1)) #method axis, score axis
        curAUCs = np.zeros((12,8,1)) #norm axis, method axis, score axis

        #extract comparison scores
        for n, row in enumerate(comps_rows[:-1]):
            curCOMPs[n] = [np.float(i) for i in row.split(":")[1:]]


        #get UNAVOIDS scores
        for I in range(len(ps)):
            for n, row in enumerate(unavoids_rows[(I*10)+1:(I*10)+9]):
                for m, col, in enumerate(row.split(",")[1:]):
                    curAUCs[I, n, m] = float(col)

        AUCs[Pi_ind] += curAUCs #aggregate scores for current pi for UNAVOIDS
        COMPs[Pi_ind] += curCOMPs #aggregate scores for current pi for LOFS

        progress+=1
        if progress % 50 == 0:
           print("-",progress)


    #output results to text file
    for Pi_n, Pi_str in enumerate(Pis) :
        
        fpo.write("\n\n____________________________________________________________\n")
        fpo.write("\n############################################################\n")
        fpo.write("\tPi = "+Pi_str)

        fpo.write("\n\n____________________________________________________________\n")
        fpo.write(subfolder+"\n")

        #write comparison average AUCs
        for n, row in enumerate(COMPs[Pi_n]/K[Pi_n]): #average over number of files found for each pi
            fpo.write(comps[n]+"\t")
            for col in row:
                fpo.write(str(col).ljust(20, ' ')+" ")
            fpo.write("\n")
        fpo.write("\n")

        #write UNVAOIDS average AUCs
        fpo.write("UNAVOIDS\n")
        for n, page in enumerate(AUCs[Pi_n]):
            fpo.write("p = "+ps[n]+"\n")
            for m, row in enumerate(page/K[Pi_n]): #average over number of files found for each pi
                fpo.write(methods[m])
                for col in row:
                    fpo.write(str(col).ljust(20, ' ')+" ")
                fpo.write("\n")
            fpo.write("\n")
        fpo.write("\n")

fpo.close()
fpe.close()
