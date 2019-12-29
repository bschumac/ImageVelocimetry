
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 07:34:20 2019

@author: benjamin
"""

from functions.EMD import *
import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
from statistics import mode 
from functions.hht import *


#ToDo:
# load T1 1s data
# pick center area random 10 pixel
# rund EMD on them
# find the mean frequency
#datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
#file = h5py.File(datapath+'Tb_stab_pertub20s_py_virdris_20Hz.nc','r')
#pertub = file.get("Tb_pertub")
#pertub = np.array(pertub)

datapath = datapath = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"
file = h5py.File(datapath+'Tb_stab_1Hz.nc')
#file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
pertub = file.get("Tb")
pertub = np.array(pertub)


def find_interval (data,  rec_freq = 1, plot_hht = False, outpath = "/", figname = "hht_fig"):
    
    interval_lst = []
        
    for i in range(0,9):
        rand_x = np.round(np.random.rand(),2)
        rand_y = np.round(np.random.rand(),2)
        
        x = np.round(100+(225-100)*rand_x)
        y = np.round(100+((225-100)*rand_y))
        
        if plot_hht:
            print(x)
            print(y)
        print(data.shape)
        pixel = data[:,int(x),int(y)]
        print(data.shape)
        
        act_interval = hht(data=pixel, time=np.arange(0, len(pixel)), outpath=outpath, 
                           figname=figname, freqsol=12, freqmax=12 ,timesol=int(len(data)/rec_freq), rec_freq = rec_freq, plot_hht = plot_hht)
        
        interval_lst.append(act_interval)
    
    return (mode(interval_lst))


