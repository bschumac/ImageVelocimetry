#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 21:34:46 2019

@author: benjamin
"""


from functions.EMD2d import *

from functions.BEMD import *
from functions.TST_fun import *
import matplotlib.pyplot as plt
import numpy as np
import h5py

x, y = np.arange(128), np.arange(128).reshape((-1,1))
img = np.sin(0.1*x)*np.cos(0.2*y)
emd2d = BEMD()  # BEMD() also works
IMFs_2D = emd2d(img)


plt.imshow(img)
plt.colorbar()
plt.imshow(IMFs_2D[0])
plt.colorbar()
plt.imshow(IMFs_2D[1])
plt.colorbar()


datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
file = h5py.File(datapath+'Tb_stab_pertub20s_py_virdris_20Hz.nc','r')
pertub = file.get("Tb_pertub")
pertub = np.array(pertub)

pertub = create_tst_mean(pertub,100)    

plt.imshow(pertub[244])
plt.colorbar()


test_img1 = pertub[244,75:225,75:225]
plt.imshow(test_img1)
plt.colorbar()

del(pertub)

emd2d = BEMD()  # BEMD() also works
IMFs_2D = emd2d(test_img1)

IMFs_2D.shape

plt.imshow(test_img1)
plt.colorbar()
plt.imshow(IMFs_2D[0])
plt.colorbar()
plt.imshow(IMFs_2D[1])
plt.colorbar()
plt.imshow(IMFs_2D[2])
plt.colorbar()
plt.imshow(IMFs_2D[3])
plt.colorbar()


 

outarr =np.zeros((IMFs_2D.shape[0]+1,IMFs_2D.shape[1],IMFs_2D.shape[2]))

outarr[0] =test_img1
outarr[1:19] = IMFs_2D

writeNetCDF(datapath, 'pertub244_IMF.netcdf', 'IMFs_2D', outarr)






