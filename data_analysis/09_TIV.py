#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:13:32 2019

@author: benjamin
"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import h5py
from matplotlib import colors, cm
import progressbar

datapath = "/home/benjamin/Met_ParametersTST/T0/data/"

file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
pertub = file.get("Tb_pertub")
pertub = np.array(pertub)
pertub = np.flip(pertub,1)
inter_window_size = 21
inter_window_size = int((inter_window_size-1)/2) 
max_displacement = 25



pertub= pertub[22:24]
pertub = pertub[:,20:288,0:300]


u = np.zeros((pertub.shape[1],pertub.shape[2]))
v = np.zeros((pertub.shape[1],pertub.shape[2]))

iterx = max_displacement+inter_window_size -1
itery = max_displacement+inter_window_size -1

bar4 = progressbar.ProgressBar(maxval=len(range(0,(pertub.shape[1]))), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
bar4.start()
bar4_iterator = 0 
  
for l in range(max_displacement+inter_window_size,(pertub.shape[1])-(max_displacement+inter_window_size)):    
    centery = l
    iterx+=1
    itery = max_displacement+inter_window_size -1
    for m in range(max_displacement+inter_window_size,(pertub.shape[2])-(max_displacement+inter_window_size)):
        centerx = m
        itery+=1
        inter_wind1 = pertub[0][centery-inter_window_size:centery+inter_window_size,centerx-inter_window_size:centerx+inter_window_size]
        
        # displace interigation window 2 by various 
        acc_dif_lst = []
        acc_iter_lst = []
        for i in range(-max_displacement,max_displacement):
            disp_centerx = centerx+i
            for k in range(-max_displacement,max_displacement):
                    disp_centery = centery+k
                    inter_wind2 = pertub[1][disp_centery-inter_window_size:disp_centery+inter_window_size,disp_centerx-inter_window_size:disp_centerx+inter_window_size]
                    dif = inter_wind1 - inter_wind2
                    acc_dif = sum(abs(sum(dif)))
                    acc_dif_lst.append(acc_dif)
                    acc_iter_lst.append((i,k))
                    
        
        acc_dif_lst[acc_dif_lst.index(min(acc_dif_lst))]
        #print(acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))])
        v[iterx,itery] = acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))][0]
        u[iterx,itery] = acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))][1]
    #plt.imshow(u,cmap = cm.jet, vmin=np.min(u), vmax=np.max(u))
    #plt.colorbar()
    bar4.update(bar4_iterator+1)
    bar4_iterator += 1
    
bar4.finish()


plt.imshow(u,cmap = cm.jet, vmin=np.min(u), vmax=np.max(u))
plt.colorbar()
plt.imshow(v,cmap = cm.jet, vmin=np.min(v), vmax=np.max(v))
plt.colorbar()

x = 1
plt.imshow(pertub[x],cmap = cm.gray, vmin=np.min(pertub[x]), vmax=np.max(pertub[x]))


plt.imshow(np.flipud(inter_wind1),cmap = cm.gray, vmin=np.min(inter_wind1), vmax=np.max(inter_wind1))
plt.imshow(np.flipud(inter_wind2),cmap = cm.gray, vmin=np.min(inter_wind2), vmax=np.max(inter_wind2))
plt.imshow(np.flipud(dif),cmap = cm.gray, vmin=np.min(dif), vmax=np.max(dif))