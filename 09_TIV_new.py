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
import openpiv.filters



time_a = datetime.datetime.now()

datapath = "/home/benjamin/Met_ParametersTST/T0/data/"


file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
pertub = file.get("Tb_pertub")
pertub = np.array(pertub)
pertub = np.flip(pertub,1)

uas = []
vas = []
xas = []
yas = []


window_size=24
overlap = 23
search_area_size = 36
outpath = datapath+"test_tiv/test_window_method_greyscale/"
my_dpi = 100




for i in range(0,len(pertub)-1):
    i = 13
    print(i)
    frame_a = pertub[i]
    frame_b = pertub[i+1]


    u1, v1= window_correlation_tiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                           search_area_size_x=search_area_size, search_area_size_y=0, corr_method="greyscale")
    
    
    x1, y1 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
    uas.append(u1)
    vas.append(v1)
    xas.append(x1)
    yas.append(y1)
    
    u1=np.flip(u1,0)
    v1=np.flip(v1,0)
    
    plt.figure()
    plt.imshow(frame_a, vmin = -5, vmax=5)
    plt.colorbar()
    plt.quiver(x1,y1,u1,v1*-1)
    plt.show()
    
    plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()



window_size=24
overlap = 23
search_area_size = 36

my_dpi = 100
outpath = datapath+"test_tiv/test_cc_method/"

for i in range(0,len(pertub)-1):
    print(i)
    frame_a = pertub[i]
    frame_b = pertub[i+1]
    
    u, v = cross_correlation_aiv(frame_a, frame_b, window_size=window_size, overlap=overlap,
                              dt=1, search_area_size=search_area_size, nfftx=None, 
                              nffty=None,width=2, corr_method='fft', subpixel_method='gaussian')
    
    
    x, y = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )


    #x=np.flip(x,0)
    #y=np.flip(y,0)
    u=np.flip(u,0)
    v=np.flip(v,0)
    #plt.imshow(y)
    u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)

    
    plt.figure()
    plt.imshow(frame_a, vmin = -5, vmax=5)
    plt.quiver(x,y,u,v)
    plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()





