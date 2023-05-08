"""
domain_synthesis.py

generate random tasks with different input distributions (for discrete RVs)
"""

import numpy as np
import util_io as io
import matplotlib.pyplot as plt 
from scipy.special import *
#from pdb import set_trace as bp 

def rel_entr(px,qx):
    return np.log(px / qx)

def discreteKL(px,qx):
    return np.mean(rel_entr(px,qx))

def generateData1D(Qx,n):
    samePr= len(Qx[0].shape) ==0
    data =np.zeros(n) 
    for i in range(n):
        if samePr:
            data[i]= np.random.choice(len(Qx), 1, p=Qx)
        else:
            data[i]= np.random.choice(len(Qx), 1, p=Qx[:,i])
    return data
    

def generateDistrib(nx,ny,nsamples=10000):
    [Pxy,Px,Py,X_s,Y_s] = io.generate2DSamples(nx,ny,nsamples)
    Qx = Px + 0.25*( np.random.rand(nx)-0.5)
    Qx = np.maximum(Qx,0)+0.001
    Qx = Qx / np.sum(Qx)
    print(Px,Qx)
    print('kl divergence between Px, Qx:',discreteKL(Px,Qx))

    p_YCondX =np.dot(np.linalg.inv(np.diag(Px)), Pxy).transpose()
    perturbAmp = np.arange(0,1,0.1)  

    Y_s_List = []
    Py_s_List = []
    Y_t_List = []
    Py_t_List = []
    p_YCondXList =[]

    
    X_t = generateData1D(Qx,nsamples)  
    Xs_hot = io.onehot(X_s,transform=False).transpose()
    Xt_hot = io.onehot(X_t,transform=False).transpose()
    
    for i in range(len(perturbAmp)):
        print('gen task',i,'with perturbation',perturbAmp[i],'..') 
        # control the difference of distribution between source task (X,Y) 
        # and the target task (X,Yi) through changing the conditional probabilty

        p_YiCondX = io.generatePerturbed(p_YCondX,perturbAmp[i])
        print ("P(Yi|X)",p_YCondX)
        print("validate stochastic matrix",np.sum(p_YCondX,axis=0))
        Py_s = np.dot(p_YiCondX,Px)
        Py_t = np.dot(p_YiCondX,Qx)

        Py_s_data = np.dot(p_YiCondX, Xs_hot)
        Py_t_data = np.dot(p_YiCondX,Xt_hot)

        Y_s = generateData1D(Py_s_data, nsamples ) 
        Y_t = generateData1D(Py_t_data,nsamples)

        Py_s_List.append(Py_s)
        Py_t_List.append(Py_t)
        p_YCondXList.append(p_YiCondX )        
        Y_s_List.append(Y_s)    
        Y_t_List.append(Y_t)
    return X_s,X_t, Y_s_List, Y_t_List, Px,Qx,Py_s_List, Py_t_List,p_YCondXList
                                                        

def main():
    np.random.seed(1) 
    nx = 8 # |X|
    ny = 3 # |Y|
    nsamples = 10000
    Xs,Xt, Y_s_List, Y_t_List, Px,Qx,Py_s_List, Py_t_List,p_YCondXList=  generateDistrib(nx,ny,nsamples)
    np.savetxt(Xs,Xt, Y_s_List, Y_t_List, Px,Qx,Py_s_List, Py_t_List,p_YCondXList)
    
         
if __name__ == '__main__':
    #if len(sys.argv) > 1:
    #    N = int(sys.argv[1])

    main()  
