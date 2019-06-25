#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:56:46 2019

@author: benjamin
"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import h5py
from functions.TST_fun import *
datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
file = h5py.File(datapath+'Tb_stab_virdiris_20Hz.nc','r')


tb = file.get("Tb")
tb = np.array(tb)
pertub = create_tst_pertubations_mm(tb,moving_mean_size=400)
writeNetCDF(datapath,"Tb_stab_pertub20s_py_virdris_20Hz.nc","Tb_pertub",pertub)

pertub_mean = create_tst_mean(pertub,moving_mean_size=200)
writeNetCDF(datapath,"Tb_stab_pertub20s_mean10s_py_virdris_20Hz.nc","Tb_pertub_mean",pertub_mean)