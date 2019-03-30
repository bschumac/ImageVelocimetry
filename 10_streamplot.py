#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:08:05 2019

@author: benjamin
"""
from matplotlib import pyplot as plt
import numpy as np
import h5py

datapath = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/"
out_fld_name = "streamplot/"
v_new_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/v_TIV_test_new_biggerIW_0-400.nc"
u_new_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/u_TIV_test_new_biggerIW_0-400.nc"
topo = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/Tb_pertub_cut_0-400.nc"


v_new_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/v_TIV_IW35_interval1_maxdisplacement25_400-1600.nc"
u_new_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/u_TIV_IW35_interval1_maxdisplacement25_400-1600.nc"
topo = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/Tb_pertub_cut_400-1600.nc"

file_v = h5py.File(v_new_file,'r')
file_u = h5py.File(u_new_file,'r')
file_topo = h5py.File(topo,'r')
topo = file_topo.get("Tb_pertub")
v = file_v.get("v")
v = np.array(v)
u = file_u.get("u")
u = np.array(u)

my_dpi = 100
np.min(topo)

for i in range(0,1200):
    print(i)
    fig = plt.figure()
    I = plt.imshow(topo[i], cmap = "rainbow",vmin=-5, vmax=5)
    fig.colorbar(I)
    X = np.arange(0, u[1].shape[1], 1)
    Y = np.arange(0, u[1].shape[0], 1)
    X1 = 125
    Y1 = 75
    
    U = u[i]
    V = v[i]
    speed = np.sqrt(U*U + V*V)
    
    lw = 2*speed / speed.max()
    Q = plt.streamplot(X, Y, U, V, density=0.7, color='k', linewidth=lw)
    E = plt.quiver(X1, Y1, np.mean(U)*20, np.mean(V)*20*-1, color = "red",width=0.01, headwidth=3, scale=1000)
    #fig.show() 
    plt.savefig(datapath+out_fld_name+str(i+400)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()




#### Mean wind direction streamline plot


U = np.mean(u,0)
V = np.mean(v,0)*-1
X = np.arange(0, U.shape[1], 1)
Y = np.arange(0, U.shape[0], 1)
speed = np.sqrt(U*U + V*V)

lw = 2*speed / speed.max()
plt.streamplot(X, Y, U, V, density=0.7, color='k', linewidth=lw)
plt.show()



### artificial dataset test



v_artificial = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/v_TIV_test_artificalZEROS_0-3.nc"
u_artificial = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/u_TIV_test_artificalZEROS_0-3.nc"
file_v = h5py.File(v_artificial,'r')
file_u = h5py.File(u_artificial,'r')
v = file_v.get("v")
v = np.array(v)
u = file_u.get("u")
u = np.array(u)
fig = plt.figure()
I = plt.imshow(U, cmap = "rainbow")
fig.colorbar(I)
X = np.arange(0, u.shape[1], 1)
Y = np.arange(0, u.shape[0], 1)

i = 0
U = u
V = v
speed = np.sqrt(U*U + V*V)

lw = 2*speed / speed.max()
Q = plt.streamplot(X, Y, U, V, density=2, color='k', linewidth=lw)
fig.show() 
plt.imshow(speed)
