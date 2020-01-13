#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:48:28 2020

@author: benjamin
"""



import h5py
import numpy as np
import os
import pickle

import xarray as xr 
from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='12GB')
client
import dask.array as da
filepath= '/media/benjamin/Seagate Expansion Drive/BTT_Turf/Telops_mat/'
fls = os.listdir(filepath)

filepath= '/media/benjamin/Seagate Expansion Drive/BTT_Turf/Telops_mat/'






def MatToNpy(matfld, npyfld):
    """
    Transfer .mat files (version 7.3) into numpy files making them accesable for dask arrays. 
     
    
    Parameters
    ----------
    matfld: string
        The folder path where the mat files are stored.
    npyfld: string
        The folder path where the npy files will be stored
    
    
    """
    fls = os.listdir(matfld)
    print("Files to read:")
    print(len(fls))
    counter = 0
    for file in fls:
        print(counter)
        arrays = {}
        f = h5py.File(filepath+file)
        for k, v in f.items():
            arrays[k] = np.array(v)
        
        file = file.replace(".mat", "")
        arr = arrays["Data_im"]
        np.save(npyfld+file, arr)
        counter +=1



name = os.listdir("/mnt/Seagate_Drive1/BTT_Turf/Telops_npy/")
arr = np.load("/mnt/Seagate_Drive1/BTT_Turf/Telops_npy/"+name[0])
arr.dtype
def to_npy_info(dirname, dtype, chunks, axis):
    with open(os.path.join(dirname, 'info'), 'wb') as f:
        pickle.dump({'chunks': chunks, 'dtype': dtype, 'axis': axis}, f)

to_npy_info("/mnt/Seagate_Drive1/BTT_Turf/Telops_npy/",dtype=np.dtype('float32'), 
            chunks= ((7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986,7986), (640,), (480,)),axis= 0)




btt_tb = da.from_npy_stack("/mnt/Seagate_Drive1/BTT_Turf/Telops_npy/")

btt_tb1hz = btt_tb[1::10]

c_btt_h = np.swapaxes(btt_tb1hz, 1,2)
c_btt_h = np.fliplr(btt_tb1hz)
c_btt_h.compute_chunk_sizes()



xrarr = xr.DataArray(c_btt_h)
arr.chunks()
xrarr.to_netcdf("/mnt/Seagate_Drive1/BTT_Turf/test.nc", engine = "netcdf4")
                

import matplotlib.pyplot as plt
plt.imshow(c_btt_h[0])

from functions.TST_fun import writeNetCDF, create_tst_pertubations_mm
writeNetCDF("/home/benjamin/Met_ParametersTST/", 'BTT_Tb6Hz.netcdf', 'Tb', c_btt_h)

btt_pertub = create_tst_pertubations_mm(c_btt_h,60)
writeNetCDF("/home/benjamin/Met_ParametersTST/", 'BTT_Tb1Hz.netcdf', 'Tb_pertub', btt_pertub)

