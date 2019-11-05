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
from joblib import Parallel, delayed
import multiprocessing
from TST_fun import *
from scipy import stats
import datetime
from math import sqrt



time_a = datetime.datetime.now()

datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
outpath = datapath+"test_tiv/test_window_method/"

file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
pertub = file.get("Tb_pertub")
pertub = np.array(pertub)
pertub = np.flip(pertub,1)

uas = []
vas = []
xas = []
yas = []


for i in range(0,len(pertub)-1):
    print(i)
    frame_a = pertub[i]
    frame_b = pertub[i+1]


    u1, v1= window_correlation_aiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                           search_area_size_x=search_area_size, search_area_size_y=0, corr_method="ssim")
    
    
    x1, y1 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
    uas.append(u1)
    vas.append(v1)
    xas.append(x1)
    yas.append(y1)
    plt.figure()
    plt.imshow(frame_a)
    plt.quiver(x1,y1,u1,v1)
    #plt.show()
    
    plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()
