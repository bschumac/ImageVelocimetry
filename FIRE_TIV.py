#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:27:01 2019

@author: benjamin
"""



import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import h5py
from matplotlib import colors, cm
import progressbar
from joblib import Parallel, delayed
import multiprocessing
from functions.TST_fun import *
from scipy import stats
import datetime
from math import sqrt
import os
import copy
from scipy.ndimage.filters import gaussian_filter as gf



experiment = "fire1"


if experiment == "T0":
    datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
elif experiment == "T1":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
    file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
elif experiment == "fire1":
    datapath = "/data/FIRE/data"
    file = h5py.File(datapath+'Tb_stab_27Hz.nc','r')


pertub = file.get("Tb")
pertub = np.array(pertub)

pertub_mm = create_tst_mean(pertub,27)

method = "greyscale"
my_dpi = 300

ws=16
ol = 15
sa = 24
olsa = 20

interval = 1

outpath = datapath+"tiv/method_"+method+"_WS_"+str(ws)+"_OL_"+str(ol)+"_SA_"+str(sa)+"_SAOL_"+str(olsa)+"/"

if not os.path.exists(outpath):
    os.makedirs(outpath)
 

#pertub[0,220:235,220:225] = 25
#pertub[6,220:235,220+15:225+15] = 25

#del(pertub)

plt.imshow(pertub[interval])
#len(pertub_mm)-interval
#pertub.shape[0]-6
for i in range(0, pertub.shape[0]-6):
    if i % 10 == 0:
        print(i)
    u, v= window_correlation_tiv(frame_a=pertub_mm[i], frame_b=pertub_mm[i+5], window_size_x=ws, 
                                 overlap_window=ol, overlap_search_area=olsa, search_area_size_x=sa, 
                                 corr_method=method, mean_analysis = True, std_analysis = True, std_threshold = 15 )
    x, y = get_coordinates( image_size=pertub_mm[i].shape, window_size=sa, overlap=olsa )      
    
    u = remove_outliers(u)
    v = remove_outliers(v)
    #plt.figure()
    #streamplot(np.flipud(u),np.flipud(v*-1),x,y,topo=pertub[i],enhancement = 1000,vmin = 0, vmax = 500, den=3.5, lw=0.7)
    #plt.savefig(outpath+str(i)+"streamplt.png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    #plt.close()
    #plt.imshow(gf(u,0.25))
    #plt.colorbar()
    
    
    my_dpi=300
    plt.imshow(pertub[i], vmin = 0, vmax=500)
    plt.colorbar()
    plt.quiver(x,y,np.flipud(np.round(u,2)),np.flipud(np.round(v,2)))
    plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()
    
    
    v= np.reshape(v,((1,v.shape[0],v.shape[1])))
    u= np.reshape(u,((1,u.shape[0],u.shape[1])))
    
    if i == 0:
        uas =  copy.copy(u)
        vas =  copy.copy(v)

    else:
        uas = np.append(uas,u,0)
        vas = np.append(vas,v,0)
    
writeNetCDF(datapath,"UAS.nc","u",uas)
writeNetCDF(datapath,"VAS.nc","v",vas)



