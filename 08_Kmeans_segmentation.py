#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:04:38 2018

@author: benjamin
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import h5py
from matplotlib import colors, cm
from functions.TST_fun import tst_kmeans

n_clusters = 8
my_dpi = 100


datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"


file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
tb= file.get("Tb_pertub")
tb = pertub = np.array(tb)
labels= []
centers = []




 


def tst_kmeans(dataset,n_clusters = 8, outpath="", my_dpi = 100):
    kmean_arr = copy.copy(dataset)
    labels= []
    centers = []
    for i in range(0,len(dataset)):
        print(i)
        
              
        arr= dataset[i]
        
        original_shape = arr.shape # so we can reshape the labels later
        
        samples = np.column_stack([arr.flatten()])
        
        clf = sklearn.cluster.KMeans(n_clusters=n_clusters)
        lab = clf.fit_predict(samples).reshape(original_shape)
        labels.append(lab)
        centers.append(clf.cluster_centers_)
     
    
    
    baseline_center = centers[0]
    mean_centers = []
    
    minarray = np.zeros((n_clusters,2))
    
    for i in range(0,len(centers[0])):
        minarray[i,0] = i
        act_number = baseline_center[i]
        collected_numbers=[]
        for center_arr in centers[1:]:     
            for k in range(0,len(centers[0])):
                minarray[k,1] = abs(center_arr[k]-act_number)
            
            #print(center_arr[(np.argmin(minarray[:,1]))])
            #print(min(minarray[:,1]))
            collected_numbers.append(center_arr[(np.argmin(minarray[:,1]))])
        mean_centers.append(np.mean(collected_numbers))
    
    
    b = np.zeros((n_clusters,1))
    b[:,0] = mean_centers            
    
    
    for i in range(0,len(dataset)):
        print(i)
        
             
        arr= dataset[i]
        
        original_shape = arr.shape # so we can reshape the labels later
        
        samples = np.column_stack([arr.flatten()])
        
        clf = sklearn.cluster.KMeans(n_clusters=n_clusters, init=b)
        lab = clf.fit_predict(samples).reshape(original_shape)
        lab = np.flipud(lab)
        labels.append(lab)
        kmean_arr[i] = lab
        from scipy.ndimage.filters import gaussian_filter as gf
        lab = gf(lab,1)
        if outpath:
            fig = plt.figure()
            plt.imshow(lab,interpolation=None, cmap = cm.gray)
            plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
            plt.close()
    return(kmean_arr)
        






dataset = tb[1:4,:,:]


test_arr = tst_kmeans(tb[0:30,:,:],outpath="/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/kmeans/", my_dpi=300, n_clusters = 8)



fig = plt.figure()
plt.imshow(test, cmap = cm.gray)
plt.savefig(outpath+str(i+35)+".png",dpi=300,bbox_inches='tight',pad_inches = 0,transparent=False)
plt.close()

