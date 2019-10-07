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

from functions.hht import *
#ToDo:
# load T1 1s data
# pick center area random 10 pixel
# rund EMD on them
# find the mean frequency
datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
file = h5py.File(datapath+'Tb_stab_pertub20s_py_virdris_20Hz.nc','r')
pertub = file.get("Tb_pertub")
pertub = np.array(pertub)

datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
pertub = file.get("Tb_pertub")
pertub = np.array(pertub)



rand_x = np.round(np.random.rand(),2)
print(np.round(rand_x,2))
(225-100)*x
rand_y = np.round(np.random.rand(),2)

y = 100+((225-100)*rand_y)

pertub.shape


s = pertub[:,220,150]

hht(data=s, time=np.arange(0, len(s)), outpath="/home/benjamin/Met_ParametersTST/T1/", figname="hhttest1", freqsol=20, timesol=20)

emd = EMD()
IMFs = emd(s)
hht0 = scipy.signal.hilbert(IMFs[0])


s.shape
plt.plot(s)
plt.plot(IMFs[0])
plt.plot(hht0)
