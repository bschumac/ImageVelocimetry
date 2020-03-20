#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:48:28 2020

@author: benjamin
"""

#from dask.distributed import Client, progress
#client = Client(processes=False, threads_per_worker=4,
#                n_workers=1, memory_limit='12GB')
#client

import h5py
import numpy as np
import os
import pickle
import xarray as xr 
import dask.array as da
import matplotlib.pyplot as plt
from functions.TST_fun import to_npy_info, writeNetCDF


# One time deal:
# run to convert mat to numpy and write info file

#matfld= '/media/benjamin/Seagate Expansion Drive/BTT_Turf/Telops_mat/'
#npyfld= '/media/benjamin/Seagate Expansion Drive/BTT_Turf/Telops_npy/'

#MatToNpy(matfld,npyfld)

#to_npy_info("/mnt/Seagate_Drive1/BTT_Turf/Telops_npy/",dtype=np.dtype('float32'), 
#            chunks= ((7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986), (640,), (480,)),axis= 0)




btt_tb = da.from_npy_stack("/mnt/Seagate_Drive1/BTT_Turf/Telops_npy/")

#btt_tb_std = da.nanstd(btt_tb,0)
#std_test= np.array(btt_tb_std)
#std_test = btt_tb_std.compute()

#plt.imshow(btt_tb_std)


btt_tb20hz = btt_tb[1::3]


c_btt_h = np.swapaxes(btt_tb6hz, 1,2)
c_btt_h = np.fliplr(btt_tb6hz)
c_btt_h.compute_chunk_sizes()

btt_tb.to_netcdf("/mnt/Seagate_Drive1/BTT_Turf/BTT_Tb_Hz.nc", engine = "netcdf4")

xrarr = xr.DataArray(c_btt_h, dims=['time', 'x', 'y'])
rolling = xrarr.rolling({'time':10}, center=True)

rolling_mean = rolling.mean()

pertub = c_btt_h - rolling_mean

pertub = np.swapaxes(pertub, 1,2)

pertub = np.flip(pertub, 1) 
plt.imshow(pertub[10])
pertub = np.fliplr(pertub)
pertub = xr.DataArray(pertub, dims=['time', 'x', 'y'])
pertub.to_netcdf("/mnt/Seagate_Drive1/BTT_Turf/pertub_test.nc", engine = "netcdf4")

#writeNetCDF("/mnt/Seagate_Drive1/BTT_Turf/","pertub_test.nc","pertub",pertub_np)


