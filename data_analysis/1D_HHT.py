
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 07:34:20 2019

@author: benjamin
"""

from functions.TST_fun import find_interval, create_tst_mean
import h5py
import numpy as np
import matplotlib.pyplot as plt


#ToDo:
# load T1 1s data
# pick center area random 10 pixel
# rund EMD on them
# find the mean frequency

datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
pertub = file.get("Tb_pertub")
pertub = np.array(pertub)
#pertub = create_tst_mean(pertub,4)

datapath = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"
file = h5py.File('/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/Tb_stab_27Hz.nc','r')
#file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
tb = file.get("Tb")
tb = np.array(tb)


#test_data= readnetcdftoarr("/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/Tb_3Hz_pertub_60s_py_mean3s.nc", "Tb_pertub")



ret_lst = find_interval(data = pertub,rec_freq = 27, plot_hht=True, outpath="/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/Test/" )