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
fls = os.listdir(filepath)
counter = 0
for file in fls:
    print(counter)
    arrays = {}
    f = h5py.File(filepath+file)
    for k, v in f.items():
        arrays[k] = np.array(v)
    
    file = file.replace(".mat", "")
    arr = arrays["Data_im"]
    np.save("/media/benjamin/Seagate Expansion Drive/BTT_Turf/Telops_npy/"+file, arr)
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

btt_tb1hz = btt_tb[1::60]
c_btt= btt_tb1hz.compute()


writeNetCDF("/mnt/Seagate_Drive1/BTT_Turf/", 'BTT_Tb1Hz.netcdf', 'Tb', c_btt)


from itertools import product, zip_longest
mmap_mode="r"

name = "from-npy-stack-%s" % dirname
keys = list(product([name], *[range(len(c)) for c in chunks]))
values = [
    (np.load, os.path.join(dirname, "%d.npy" % i), mmap_mode)
    for i in range(len(chunks[axis]))
]
dsk = dict(zip(keys, values))








from functions.TST_fun import create_tst_subsample_mean, create_tst_subsample

btt_tb1hz = create_tst_subsample(btt_tb,size=60)



Tb = file['Tb_streamwise_Area_EE3']


arr3 = xr.DataArray(np.random.randn(2, 3),[('x', ['a', 'b']), ('y', [10, 20, 30])])

