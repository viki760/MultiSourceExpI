import numpy as np
import ot
import geomloss
import torch
import math

def compute_coupling(X_src, X_tar):
    
    cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)

    C = cost_function(X_src,X_tar).numpy()
    P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tar.shape[0]), C, numItermax=100000)
    W = np.sum(P*np.array(C))

    return P,W


def compute_CE(P, Y_src, Y_tar):
    src_label_set = set(sorted(list(Y_src.flatten())))
    tar_label_set = set(sorted(list(Y_tar.flatten())))

    # joint distribution of source and target label
    P_src_tar = np.zeros((np.max(Y_src)+1,np.max(Y_tar)+1))

    for y1 in src_label_set:
        y1_idx = np.where(Y_src==y1)
        for y2 in tar_label_set:
            y2_idx = np.where(Y_tar==y2)

            RR = y1_idx[0].repeat(y2_idx[0].shape[0])
            CC = np.tile(y2_idx[0], y1_idx[0].shape[0])

            P_src_tar[y1,y2] = np.sum(P[RR,CC])

    # marginal distribution of source label
    P_src = np.sum(P_src_tar,axis=1)

    ce = 0.0
    for y1 in src_label_set:
        P_y1 = P_src[y1]
        for y2 in tar_label_set:
            
            if P_src_tar[y1,y2] != 0:
                ce += -(P_src_tar[y1,y2] * math.log(P_src_tar[y1,y2] / P_y1))
    return ce

def OTCE(data_s, data):
            
    src_x, src_y = next(iter(data_s))
    tar_x, tar_y = next(iter(data))
    
    # obtain the optimal coupling matrix P and the wasserstein distance W
    P,W = compute_coupling(src_x.reshape([len(src_x), -1]), tar_x.reshape([len(tar_x), -1]))

    # compute the conditonal entropy (ce)
    ce = compute_CE(P, np.array(src_y), np.array(tar_y))
    print ('Wasserstein distance:%.4f, Conditonal Entropy: %.4f'%(W,ce))

    return W, ce