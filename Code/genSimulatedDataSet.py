#this script generates a simulated dataset
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load



def createDataSet():
    data = [] 

    #create first two dimensions as a grid
    for i in np.arange(start=0.0, stop=0.201, step=0.01): #bottom left corner
        for j in np.arange(start=0.0, stop=0.201, step=0.01):
            data.append(np.array([i,j]))

    for i in np.arange(start=0.8, stop=1.01, step=0.02): # top right corner
        for j in np.arange(start=0.8, stop=1.01, step=0.02):
            data.append(np.array([i,j]))

    #append outliers
    data = np.append(data,np.reshape([[0.22,0.22],[0.105,0.105],[0.5,0.5]], (3,2)), axis=0)

    #save first two dimensions to csv
    fp = open("2dGrid.csv","w")
    fp.write("x1,x2\n")
    for i in data:
        fp.write(str( round(i[0], 2) )+","+str( round(i[1], 2) )+"\n")
    fp.close()

    tBL = 441 #how many points to sample from bottom left cluster
    tTR = 121 #how many points to sample from top right cluster

    BL = [] #bottom left cluster distribution
    for i in np.arange(0, 0.20, 0.01):
        BL.append(i)

    TR = [] #top right cluster distribution
    for i in np.arange(0.80, 1.01, 0.02):
        TR.append(round(i,2))

    for i in range(510):
        BLdata = np.reshape(np.random.choice(BL, tBL), (tBL,1)) #sample tBL point from Bottom Left cluster
        TRdata = np.reshape(np.random.choice(TR, tTR), (tTR,1)) #sample tTR point from Top Right cluster

        newcol = np.append(BLdata, TRdata, axis=0) #append clusters together
        newcol = np.append(newcol,np.reshape([0.22,0.105,0.5], (3,1)), axis=0) #append outliers

        data = np.append(data, newcol, axis=1) #append (2+i)th column to dataset

    return data

def drawDataSet(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(data[:,0],data[:,1], s=0.75, c="grey") #plot clusters
    ax.scatter(data[data.shape[0]-1,0],data[data.shape[0]-1,1], s=1.5, c="red") #plot 50/50 outlier
    ax.scatter(data[data.shape[0]-2,0],data[data.shape[0]-2,1], s=1.5, c="green") #plot bottom left edge outlier
    ax.scatter(data[data.shape[0]-3,0],data[data.shape[0]-3,1], s=1.5, c="blue") #plot bottom left inner outliter

    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop)))

    plt.show()
    #plt.savefig("dataset.png",pad_inches=0.1, dpi=250, bbox_inches='tight')
    plt.close()



data = createDataSet()
drawDataSet(data)

dump(data, "simData.joblib") #save dataset
#data = load("data.joblib")[:,:2]

#fp = open("data.csv", "w+")
#for n, row in enumerate(data):
#    fp.write(str(row[0])+","+str(row[1])+"\n")






