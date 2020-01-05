#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:13:32 2019

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





experiment = "pre_fire_1Hz"


if experiment == "T0":
    datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
elif experiment == "T1":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
    file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')
    mean_time = 5
elif experiment == "T120Hz":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
    file = h5py.File(datapath+'Tb_stab_pertub20s_py_virdris_20Hz.nc','r')
    mean_time = 0
elif experiment == "pre_fire_27Hz":
    datapath = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"
    file = h5py.File(datapath+'Tb_stab_cut_27Hz_noscale_norot.nc')
    mean_time = 3


tb = file.get("Tb")
tb = np.array(tb)


tb = create_tst_subsample(tb, 9)

#tb = create_tst_subsample_mean(tb, 27)

if mean_time != 0:
    tb = create_tst_mean(tb,mean_time) 


pertub = create_tst_pertubations_mm(tb,180)




#writeNetCDF(datapath, "Tb_3Hz_pertub_60s_py_cut_norot.nc", "Tb_pertub", pertub)





method = "greyscale"
my_dpi = 300
time_interval = 1


ws=16
ol = 15
sa = 32
olsa = 28
mean_a = False
std_a = False

outpath = (datapath+"tiv/experiment_"+experiment+"_meantime_"+str(mean_time)+"_interval_"+str(time_interval)+"_method_"+method+"_WS_"+
str(ws)+"_OL_"+str(ol)+"_SA_"+str(sa)+"_SAOL_"+str(olsa)+"_mean_a_"+str(mean_a)+"_std_a"+str(std_a)+"/")

if not os.path.exists(outpath):
    os.makedirs(outpath)
        
print(outpath)



for i in range(180, len(pertub)-time_interval):
    
    print(i)
    u, v= window_correlation_tiv(frame_a=pertub[i], frame_b=pertub[i+time_interval], window_size_x=ws, overlap_window=ol, overlap_search_area=olsa, 
                                 search_area_size_x=sa, corr_method=method, mean_analysis = mean_a, std_analysis = std_a, std_threshold = 10)
    x, y = get_coordinates( image_size=pertub[i].shape, window_size=sa, overlap=olsa )      
    
    u = remove_outliers(u,filter_size=9, sigma=1)
    v = remove_outliers(v, filter_size=9, sigma=1)
    
    
    plt.imshow(pertub[i], vmin = -1, vmax=1, cmap = "gist_rainbow_r")
    plt.colorbar()
    plt.quiver(x,y,np.flipud(np.round(u,2)),np.flipud(np.round(v,2)))
    plt.savefig(outpath+'{:04d}'.format(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()
    
    
    #plt.imshow(u)
    
    v= np.reshape(v,((1,v.shape[0],v.shape[1])))
    u= np.reshape(u,((1,u.shape[0],u.shape[1])))

    if i == 180:
        uas =  copy.copy(u)
        vas =  copy.copy(v)

    else:
        uas = np.append(uas,u,0)
        vas = np.append(vas,v,0)


writeNetCDF(outpath, 'UAS.netcdf', 'u', uas)

writeNetCDF(outpath, 'VAS.netcdf', 'v', vas)

writeNetCDF(outpath, 'WS.netcdf', 'ws', calcwindspeed(uas,vas)*0.2)

writeNetCDF(outpath, 'WD.netcdf', 'wd', calcwinddirection(uas,vas))


  
writeNetCDF(outpath, 'Tb_stab_pertub_mean.netcdf', 'Tb_pertub20Hz_means', pertub)





#pertub[0] = 0
#pertub[6] = 0
#pertub[0,145:154,145:160] = 25
#pertub[0,15:25,15:25] = 25
#pertub[0,215:225,215:225] = 25
#pertub[0,25:55,315:330] = 25
#
#
#pertub[time_interval,145+12:154+12,145+12:160+12] = 25
#pertub[time_interval,15+12:25+12,15+12:25+12] = 25
#pertub[time_interval,215+12:225+12,215+12:225+12] = 25
#pertub[time_interval,25+20:55+20,315+20:330+20] = 25



#pertub[0,220:235,220:225] = 25
#pertub[6,220:235,220+15:225+15] = 25



#plt.imshow(pertub[10]-pertub[10+1])
#pertub[0].shape
#len(pertub)-time_interval)

   



plt.imshow(np.flipud(v)-np.flipud(u))
calcwinddirection(1,1)


# checked and trusted!!!
# v immer *-1 dann passt das
# fuer quiver plot  dann flipud 

plt.imshow(calcwindspeed(u,v))
plt.imshow(calcwinddirection(u,v))
plt.colorbar()
np.isnan(u)
np.round(calcwinddirection(u,v*-1)[~numpy.isnan(calcwinddirection(u,v*-1))],0)
np.round((u)[~numpy.isnan(u)],2)

v.shape
calcwinddirection(1,0.5)
plt.figure()
streamplot(np.flipud(u),np.flipud(v),x,y,topo=pertub[i],enhancement = 1000,vmin = -15, vmax = 15, den=2, lw=2)
plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
plt.close()



u = np.flipud(u)
v = np.flipud(v)

my_dpi=1200
plt.imshow(pertub[0], vmin = vmin, vmax=vmax)
plt.quiver(x,y,np.flipud(np.round(u,0)),np.flipud(np.round(v,0)))
plt.savefig(outpath+str(i+5)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
plt.close()

def quiverplot(U, V, X, Y, enhancement = 2, topo = None, vmin = -2, vmax = 2, cmap = "gist_rainbow", lw="ws", den = 1)
    
    if topo is None:
        topo = copy.copy(u)
    fig = plt.figure()
    plt.imshow(pertub[0], vmin = vmin, vmax=vmax)
    plt.quiver(x,y,np.flipud(np.round(u,2)),np.flipud(np.round(v,2)))
    return(fig)
    










if method == "greyscale" or method =="rmse" or method =="ssim":

    for i in range(47,len(pertub)-1,1):
        i = 0
        print(i)
        frame_a = pertub[i]
        frame_b = pertub[i+1]
    
    
        u1, v1= window_correlation_tiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                               search_area_size_x=search_area_size, search_area_size_y=0, corr_method=method)
        
        
        x1, y1 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
        
        
        
        

        u1=np.flipud(u1)
        v1=np.flipud(v1)
        
        
        plt.figure(1)
        #plt.subplot(211)
        #plt.gca().set_title("IV")
        plt.imshow(frame_a,vmin=0,vmax=7)
        plt.colorbar()
        plt.quiver(x1,y1,u1,v1,color="black")        

        #streamplot(U=u1, V=v1*-1, X=x1, Y=y1, topo = frame_a,vmin=-1,vmax=1)
       
        #plt.show()
        plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
        plt.close()
        
        v1= np.reshape(v1,((1,v1.shape[0],v1.shape[1])))
        u1= np.reshape(u1,((1,u1.shape[0],u1.shape[1])))
        if i == 139 or (i-139)//337 > lasttime:
            if i != 139:
                uas_lst.append(uas)
                vas_lst.append(vas)
            uas =  copy.copy(u1)
            vas = copy.copy(v1)
            pertub_minmean = np.mean(pertub[i:i+337,:,:],axis=0)
            pertub_minmean_lst.append(pertub_minmean)
            lasttime = (i-139)//337
            
            print(lasttime)
        else:
            uas = np.append(uas,u1,0)
            vas = np.append(vas,v1,0)
        if i == 139:
            uas2 =  copy.copy(u1)
            vas2 = copy.copy(v1)

        else:
            uas2 = np.append(uas,u1,0)
            vas2 = np.append(vas,v1,0)
            
    
    
else:    
    for i in range(139,len(pertub)-6,6):
        print(i)
        frame_a = pertub[i]
        frame_b = pertub[i+6]
        
        u, v = cross_correlation_aiv(frame_a, frame_b, window_size=window_size, overlap=overlap,
                                  dt=1, search_area_size=search_area_size, nfftx=None, 
                                  nffty=None,width=2, corr_method='fft', subpixel_method='gaussian')
        
        
        x, y = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )

        ## vectorplot
        #u = np.flipud(u)
        #v = np.flipud(v)
        #plt.imshow(y)
        #v = v*-1
        #u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    
        
        #plt.figure()
       # plt.imshow(frame_a, vmin = -2, vmax=2)
        #plt.colorbar()
       # plt.quiver(x,y,u,v)
        #plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
        #plt.close()









x1 = x1 - 2

pertub_mean.shape

    
len(uas_lst)
outpath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/tiv/method_streamplot_minmean_rmse_WS_24_OL_23_SA_48/"

if not os.path.exists(outpath):
    os.makedirs(outpath)
        

for i in range (0, len(uas_lst)):
    print(i)        
    uas_mean = np.nanmean(uas_lst[i],axis=0)        
           
    vas_mean=np.nanmean(vas_lst[i],axis=0)        
    pertub_mean = np.nanmean(pertub,axis=0)       
    
    X= x1
    Y= y1
    U = uas_mean
    V = vas_mean*-1
    topo = pertub_minmean_lst[i]
    if topo is None:
        topo = copy.copy(U)
    fig = plt.figure()
    I = plt.imshow(topo, cmap = "rainbow", vmin = -0.2, vmax=0.2)
    fig.colorbar(I)

    speed = np.sqrt(U*U + V*V) 
    
    
    lw = 3*(speed / speed.max())
    
    
    lw[(lw< np.mean(lw)+3*np.std(lw)) & (lw> np.mean(lw)-2*np.std(lw)) ] =  lw[(lw< np.mean(lw)+3*np.std(lw)) & (lw> np.mean(lw)-2*np.std(lw)) ]*5
    Q = plt.streamplot(X, Y, U, V, color='k', linewidth=lw)
    #plt.savefig(outpath+str("10_min_mean")+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    #plt.close()  
    #plt.show()
    
    plt.savefig(outpath+str("min_mean_")+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()        
        
        
        
        
#def streamplot(U, V, X, Y, topo = None, vmin = -2, vmax = 2):
    uas_mean = np.nanmean(uas2,axis=0)        
           
    vas_mean=np.nanmean(vas2,axis=0)        
    pertub_mean = np.nanmean(pertub,axis=0)     

    X= x1
    Y=y1
    U = uas_mean
    V = vas_mean*-1
    topo = pertub_mean
    if topo is None:
        topo = copy.copy(U)
    fig = plt.figure()
    I = plt.imshow(topo, cmap = "rainbow", vmin = -0.02, vmax=0.02)
    fig.colorbar(I)

    speed = np.sqrt(U*U + V*V)   
    lw = 2*speed / speed.max()
    Q = plt.streamplot(X, Y, U, V)
    plt.savefig(outpath+str("10_min_mean")+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()  
        
        


