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

import matplotlib.pyplot as plt
from scipy import signal



### USER INPUT ###

experiment = "BTT"




### END ###



if experiment == "T0":
    datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
elif experiment == "T1":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
    file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
    mean_time = 5
    
elif experiment == "T120Hz":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/"
    file = h5py.File(datapath+'Optris_data/Tb_stab_cut_red_20Hz.nc','r')
    outpath = datapath+"energy_spectrum/"
    mean_time = 0
    fs = 20
elif experiment == "pre_fire_27Hz":
    datapath = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"
    outpath = datapath+"energy_spectrum/"
    file = h5py.File(datapath+'Tb_stab_cut_red_27Hz.nc')
    fs = 27
    mean_time = 0
    subsample = 9
    hard_subsample = False
    
elif experiment == "BTT":
    datapath = "/mnt/Seagate_Drive1/BTT_Turf/"
    outpath = datapath
    tb = np.load(datapath+'Telops_npy/3.npy')
    fs = 20


if not os.path.exists(outpath):
    os.makedirs(outpath)
   

tb = file.get("Tb")

tb = np.array(tb)

tb = tb[1::3]


tb_cut = tb[:,200:500,200:500]
pix = tb_cut[:,5,5]

plt.plot(final_list)

def outlier_removal(arr):
    
    mean = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0)
    
    final_list = [x for x in arr if (x > mean - 2 * sd)]
    return(final_list)
    


#x = pix - np.mean(pix)


#tb_cut.flatten()

fq_lst = []
pxx_lst = []


for i in range(0,len(c[1,:,:])):
    print(i)
    for j in range(0,len(tb_cut[1,:,:])):
        act_pixx = tb_cut[:,i,j]
        act_pixx = outlier_removal(act_pixx)
        f, Pxx = signal.welch(act_pixx, fs, nperseg=1024)
        fq_lst.append(f)
        pxx_lst.append(Pxx*f)
    




a = np.array([0.03,0.1,1])

b = np.power(10,np.log10(a)*(-1*5/3)-2.25)




plt.yscale("log")
plt.xscale("log")
#plt.plot(fq_lst[20],pxx_lst[20])
plt.plot(fq_lst[185],pxx_lst[185])
plt.plot(fq_lst[200],pxx_lst[200])
plt.plot(fq_lst[266],pxx_lst[266])
#plt.plot(np.mean(fq_lst),np.mean(pxx_lst))


my_dpi = 300
plt.plot(a,b)
plt.savefig(outpath+"BTT_Turf_20Hz_energy_spectrum.png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
plt.close()































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