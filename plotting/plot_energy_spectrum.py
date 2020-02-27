#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:38:38 2020

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
#import openpiv.filters
import os
import copy
import scipy
import pathos
import skimage



### USER INPUT ###

experiment = "T120Hz"




### END ###



if experiment == "T0":
    datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
elif experiment == "T1":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
    file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
    mean_time = 5
    
elif experiment == "T120Hz":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
    file = h5py.File(datapath+'Tb_stab_pertub20s_py_virdris_20Hz.nc','r')
    mean_time = 0
    fs = 20
elif experiment == "pre_fire_27Hz":
    datapath = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"
    file = h5py.File(datapath+'Tb_stab_cut_red_27Hz.nc')
    fs = 27
    mean_time = 0
    subsample = 9
    hard_subsample = False
    




tb = file.get("Tb_pertub")
tb = np.array(tb)


tb_cut = tb[:,50:250,50:250]
pix = tb_cut[:,100,100]


plt.imshow(tb[1,:,:])


import matplotlib.pyplot as plt
from scipy import signal


#x = pix - np.mean(pix)


#tb_cut.flatten()

fq_lst = []
pxx_lst = []


for i in range(0,len(tb_cut[1,:,:])):
    print(i)
    for j in range(0,len(tb_cut[1,:,:])):
        f, Pxx = signal.welch(tb_cut[:,i,j], fs, nperseg=1024)
        fq_lst.append(f)
        pxx_lst.append(Pxx*f)
    




a = np.array([0.03,0.1,1])

b = np.power(10,np.log10(a)*(-1*5/3)-3.5)




plt.yscale("log")
plt.xscale("log")
#plt.plot(fq_lst[1],pxx_lst[1])
#plt.plot(fq_lst[20],pxx_lst[20])
plt.plot(fq_lst[173],pxx_lst[173])
plt.plot(fq_lst[54],pxx_lst[54])
#plt.plot(np.mean(fq_lst),np.mean(pxx_lst))



plt.plot(a,b)
plt.show()































fq_arr = np.array(fq_lst)
pxx_arr = np.array(pxx_lst)
fq_mean = np.mean(fq_arr,axis=1)
pxx_mean =np.mean(pxx_arr,axis=0)

#f_res, Pxx_res = signal.welch(x, fs, nperseg=2048)





plt.yscale("log")
plt.xscale("log")
plt.plot(fq_lst[1],pxx_mean*fq_lst[1])









plt.subplot(3,1,1)
plt.plot(x)

plt.subplot(3,1,2)
plt.plot(f, Pxx)
plt.xlim([0, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')

plt.subplot(3,1,3)
plt.plot(f_res, Pxx_res)
plt.xlim([0, 1])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')

plt.show()

Hn = np.fft.fft()










freqs = np.fft.fftfreq(len(Hn), 1/fs)
idx = np.argmax(np.abs(Hn))
freq_in_hertz = freqs[idx]
print('Main freq:'+ str(freq_in_hertz))
print ('RMS amp:' + str(np.sqrt(Pxx.max())))