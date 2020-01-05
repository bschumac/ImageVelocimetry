#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:27:01 2019

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
import os
import copy
from scipy.ndimage.filters import gaussian_filter as gf

from scipy import ndimage



experiment = "fire0ee3"


if experiment == "T0":
    datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
    prefix = "T0"
elif experiment == "T1":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
    file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
    prefix = "T1"
elif experiment == "fire1_P1":
    datapath = "/data/FIRE/data/"
    file = h5py.File(datapath+'Tb_stab_27Hz.nc','r')
    prefix = "P1"
elif experiment == "fire0ee1":
    datapath = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2018/"
    file = h5py.File(datapath+'Tb_streamwise_Area_EE1.nc','r')
    prefix = "EE1"
elif experiment == "fire0ee3":
    datapath = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2018/"
    file = h5py.File(datapath+'Tb_streamwise_Area_EE3.nc','r')
    prefix = "EE3"
elif experiment == "fire0ee1_99perc":
    datapath = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2018/"
    file = h5py.File(datapath+'Tb_stream_bw_Area_99th_EE1.nc','r')
    prefix = "EE1_99perc"
elif experiment == "fire0ee3_99perc":
    datapath = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2018/"
    file = h5py.File(datapath+'Tb_stream_bw_Area_99th_EE3.nc','r')
    prefix = "EE3_99perc"


pertub = file.get("Tb")
pertub = np.array(pertub)
if experiment == "fire0ee3" or experiment == "fire0ee1" or experiment == "fire0ee1_99perc" or experiment == "fire0ee3_99perc":
    pertub = np.swapaxes(pertub, 0,2)
    if experiment == "fire0ee1_99perc":
        datapath2 = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2018/"
        file2 = h5py.File(datapath+'Tb_streamwise_Area_EE1.nc','r')
        mask = file2.get("Tb")
        mask = np.array(mask)
        mask = np.swapaxes(mask, 0,2)
        pertub = mask*pertub
    elif experiment == "fire0ee3_99perc":
        datapath2 = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2018/"
        file2 = h5py.File(datapath+'Tb_streamwise_Area_EE3.nc','r')
        mask = file2.get("Tb")
        mask = np.array(mask) 
        mask = np.swapaxes(mask, 0,2)
        pertub = mask*pertub
else:
    pertub = create_tst_mean(pertub,5)

method = "greyscale"
my_dpi = 300

ws=16
ol = 15
sa = 24
olsa = 20

interval = 1

outpath = datapath+"tiv/"+prefix+"_method_"+method+"_WS_"+str(ws)+"_OL_"+str(ol)+"_SA_"+str(sa)+"_SAOL_"+str(olsa)+"/"

if not os.path.exists(outpath):
    os.makedirs(outpath)
 
    writeNetCDF(datapath, "test.nc", "Tb", pertub)

#pertub[0,220:235,220:225] = 25
#pertub[6,220:235,220+15:225+15] = 25

#del(pertub)

plt.imshow(pertub[500])
plt.colorbar()
#len(pertub_mm)-interval
#pertub.shape[0]-6

for i in range(0, pertub.shape[0]-6):
    if i % 10 == 0:
        print(i)
    u, v= window_correlation_tiv(frame_a=pertub[i], frame_b=pertub[i+1], window_size_x=ws, 
                                 overlap_window=ol, overlap_search_area=olsa, search_area_size_x=sa, 
                                 corr_method=method, mean_analysis = True, std_analysis = True, std_threshold = 15 )
    x, y = get_coordinates( image_size=pertub[i].shape, window_size=sa, overlap=olsa )      
    
    u = remove_outliers(u)
    v = remove_outliers(v)
    #plt.figure()
    #streamplot(np.flipud(u),np.flipud(v*-1),x,y,topo=pertub[i],enhancement = 1000,vmin = 0, vmax = 500, den=3.5, lw=0.7)
    #plt.savefig(outpath+str(i)+"streamplt.png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    #plt.close()
    #plt.imshow(gf(u,0.25))
    #plt.colorbar()
    
    
    my_dpi=300
    plt.imshow(pertub[i], vmin = 500, vmax=800)
    plt.colorbar()
    plt.quiver(x,y,np.flipud(np.round(u,2)),np.flipud(np.round(v,2)))
    plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()
    
    
    v= np.reshape(v,((1,v.shape[0],v.shape[1])))
    u= np.reshape(u,((1,u.shape[0],u.shape[1])))
    
    if i == 0:
        uas =  copy.copy(u)
        vas =  copy.copy(v)

    else:
        uas = np.append(uas,u,0)
        vas = np.append(vas,v,0)
    
writeNetCDF(datapath,prefix+"_UAS.nc","u",uas)
writeNetCDF(datapath,prefix+"_VAS.nc","v",vas)


#737
i = 757
u, v= window_correlation_tiv(frame_a=pertub[i], frame_b=pertub[i+1], window_size_x=ws, 
                                 overlap_window=ol, overlap_search_area=olsa, search_area_size_x=sa, 
                                 corr_method=method, mean_analysis = True, std_analysis = True, std_threshold = 15 )
x, y = get_coordinates( image_size=pertub[i].shape, window_size=sa, overlap=olsa )      

u = remove_outliers(u)
v = remove_outliers(v)

u = (u*5/60)*10
v = (v*5/60)*10

#plt.figure()
#streamplot(np.flipud(u),np.flipud(v*-1),x,y,topo=pertub[i],enhancement = 1000,vmin = 0, vmax = 500, den=3.5, lw=0.7)
#plt.savefig(outpath+str(i)+"streamplt.png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
#plt.close()
#plt.imshow(gf(u,0.25))
#plt.colorbar()
outpath = "/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2018/test/"

my_dpi=300
plt.imshow(pertub[i], vmin = 0, vmax=700, cmap="inferno" )
plt.colorbar()
plt.quiver(x,y,np.flipud(np.round(u,2)),np.flipud(np.round(v,2)))
plt.savefig(outpath+str(i+5)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
plt.close()


rotated_x = ndimage.rotate(x, 90)
rotated_y = ndimage.rotate(y, 90)
rotated_u = np.flipud(np.round(u,2))
rotated_v = np.flipud(np.round(v,2))

rotated_pertub = ndimage.rotate(pertub[i], 90)
rotated_pertub = ndimage.rotate(rotated_pertub, 180)

rotated_pertub = np.fliplr(np.round(rotated_pertub,2))
rotated_pertub = np.fliplr(np.round(rotated_pertub,1))

skip = (slice(None, None, 2), slice(None, None, 2))


jet_mat = np.genfromtxt("/home/benjamin/Met_ParametersTST/PNAS_Paper_Marwan/colormap_Fig2.txt", skip_header=0)
jet_mat = numpy.vstack([jet_mat, np.array((1,0,0))])

from matplotlib.colors import ListedColormap
jet_matlab = ListedColormap(jet_mat, name='jet_matlab_custom')

my_dpi=600
fig, ax = plt.subplots(figsize=(5,8))

im = ax.imshow(rotated_pertub, vmin = 0, vmax=800, cmap=jet_matlab )

Q = plt.quiver(y[skip], x[skip], v[skip],u[skip]*-1, scale_units='inches', scale=25, 
               linewidths=1.5, edgecolors='k', headwidth=5)

plt.quiverkey(Q, 1.25, 0.15, 5, r'$5 \frac{m}{s}$', labelpos='E',
                   coordinates='axes', labelsep = 0.05, angle = 0)

cax = fig.add_axes([0.75, 0.27, 0.05, 0.5])
fig.colorbar(im, cax=cax, orientation='vertical')
#plt.show()


plt.savefig(outpath+str(i)+"option_jet_0-800.png", dpi=my_dpi)
plt.close()


# , bbox_inches='tight',pad_inches = 0,transparent=False





























M = np.hypot(U, V)
Q = ax3.quiver(X, Y, U, V, M, units='x', pivot='tip', width=0.022,
               scale=1 / 0.15)

ax3.scatter(X, Y, color='0.5', s=1)

plt.show()



import matplotlib.pyplot as plt
from numpy.random import *


x, y  = [1, 2, 3], [0.5, 0.5, 0.5]
u1,v1 = randn(3), randn(3)
u2,v2 = randn(3), randn(3)
u3,v3 = randn(3), randn(3)
QV1 = plt.quiver(x, y, u1, v1, color='r')
QV2 = plt.quiver(x, y, u2, v2, color='b')
QV3 = plt.quiver(x, y, u3, v3, color='g')
plt.quiverkey(QV1, 1.2, 0.515, 2, 'arrow 1', coordinates='data')
plt.quiverkey(QV2, 1.2, 0.520, 2, 'arrow 2', coordinates='data')
plt.quiverkey(QV3, 1.2, 0.525, 2, 'arrow 3', coordinates='data')