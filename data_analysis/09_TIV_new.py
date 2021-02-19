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
from TST_fun import *
from scipy import stats
import datetime
from math import sqrt
#import openpiv.filters
import os
import copy
import scipy
import pathos
import skimage



### USER INPUT ###

experiment = "namtex"




### END ###



if experiment == "T0":
    datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
elif experiment == "T1":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/"
    outpath= "/mnt/Seagate_Drive1/T1/"
    file = h5py.File(datapath+'Tb_stab_cut_red_20Hz.nc','r')
    mean_time = 0
    subsample = 10 
    hard_subsample = False

elif experiment == "T2":
    datapath = "/mnt/Seagate_Drive1/TURF-T2/Tier02/Optris_data/Flight03/"
    outpath= "/mnt/Seagate_Drive1/TURF-T2/Tier03/Optris_data/Flight03/"
    file = h5py.File(datapath+'Tb_stab_2Hz.nc','r')
    mean_time = 0
    subsample = 0 
    hard_subsample = False



elif experiment == "T120Hz":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
    file = h5py.File(datapath+'Tb_stab_pertub20s_py_virdris_20Hz.nc','r')
    mean_time = 0
elif experiment == "pre_fire_":
    rec_freq = 27
    datapath = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"
    file = h5py.File(datapath+'Tb_stab_cut_red_27Hz.nc')
    mean_time = 0
    subsample = 9
    experiment = experiment+str(int(rec_freq/subsample))+"Hz"
    hard_subsample = False

elif experiment == "namtex":
    rec_freq = 9
    datapath = "/mnt/Seagate_Drive1/NamTEX/Tier02/"
    file = h5py.File(datapath+'9Hz_planeCorr.nc')
    mean_time = 0
    subsample = 9
    experiment = experiment+str(int(rec_freq/subsample))+"Hz"
    hard_subsample = False

    

tb = file.get("Tb")
tb = np.array(tb)








if hard_subsample:
    tb = create_tst_subsample(tb, subsample)
elif subsample != 0:
    tb = create_tst_subsample_mean(tb, subsample)
 
if mean_time != 0:
    tb = create_tst_mean(tb,mean_time) 


ret_lst = randomize_find_interval(data = tb,rec_freq = 1)







#with open(datapath+"Tb_stab_cut_red_3Hz_hardsubsample_"+str(hard_subsample)+"pixel.txt", 'w') as f:
#    for item in pixel:
#        f.write("%s\n" % item)




#pertub = create_tst_pertubations_mm(tb,120)
#writeNetCDF(datapath, "Tb120"+str(hard_subsample)+"_pertub.nc", "Tb_pertub", pertub)


#def runATIV (tb, time_interval, mean_time, outpath, ws, ol, sa, olsa, subsample, hard_subsample, pertubtime, set_len = None, method= "greyscale", my_dpi=300):
  
time_lst = [80, 60, 40, 20, 10, 5]

pertubtime = time_lst[1]
time_interval = int(ret_lst[0])
ws=16
ol = 15
sa = 32
olsa = 28
method= "greyscale"


#tb = np.where(tb < 35, tb, 0)
  
pertub = create_tst_pertubations_mm(tb,pertubtime)

outpath = (datapath+"tiv/experiment_"+experiment+"_meantime_"+str(mean_time)+"_interval_"+str(time_interval)+"_method_"+method+"_WS_"+
str(ws)+"_OL_"+str(ol)+"_SA_"+str(sa)+"_SAOL_"+str(olsa)+"_subsample_"+str(subsample)+"_hard_subsample_"+str(hard_subsample)+"_pertubtime"+str(pertubtime)+"/")

if not os.path.exists(outpath):
    os.makedirs(outpath)
        
print(outpath)



len_pertub = len(pertub)
set_len = 20
if set_len is not None:
    len_pertub = set_len
    

for i in range(60, 90-time_interval):
    #i = 60
    print(i)
#def runTIVparallel(i):   
#i = 20
    u, v= window_correlation_tiv(frame_a=pertub[i], frame_b=pertub[i+time_interval], window_size_x=ws, overlap_window=ol, overlap_search_area=olsa, 
                             search_area_size_x=sa, corr_method=method, mean_analysis = False, std_analysis = False, std_threshold = 10)
    x, y = get_coordinates( image_size=pertub[i].shape, window_size=sa, overlap=olsa )  

    
    
    plt.imshow(pertub[i], vmin = -1, vmax=1)
    plt.colorbar()
    plt.quiver(x,y,np.flipud(np.round(u,2)),np.flipud(np.round(v,2)))
    plt.savefig(outpath+'{:04d}'.format(i)+".png",dpi=300,bbox_inches='tight',pad_inches = 0,transparent=False)
    plt.close()
    v= np.reshape(v,((1,v.shape[0],v.shape[1])))
    u= np.reshape(u,((1,u.shape[0],u.shape[1])))
    if i == 60:
        uas =  copy.copy(u)
        vas =  copy.copy(v)

    else:
        uas = np.append(uas,u,0)
        vas = np.append(vas,v,0)    

u = remove_outliers(u, filter_size=3, sigma=2)
v = remove_outliers(v, filter_size=3, sigma=2)

#    return(u,v)



#uas = readnetcdftoarr("/mnt/Seagate_Drive1/T1/tiv/Assimilated_lowRes/experiment_T1_meantime_0_interval_3_method_greyscale_WS_16_OL_15_SA_32_SAOL_28_subsample_10_hard_subsample_False_pertubtime60/UAS.netcdf","u")
    

from joblib import Parallel, delayed
out_lst = Parallel(n_jobs=6)(delayed(runTIVparallel)(i) for i in range(0, len_pertub-time_interval))
out_uv = np.array(out_lst)
uas = out_uv[:,0,:,:]
vas = out_uv[:,1,:,:]

plt.imshow(out_lst[15][0], vmin=-.5, vmax=.5)



writeNetCDF(outpath, 'UAS.netcdf', 'u', uas)
writeNetCDF(outpath, 'VAS.netcdf', 'v', vas)






from joblib import Parallel, delayed
Parallel(n_jobs=6)(delayed(runATIV)(tb=tb, time_interval=time_interval, mean_time=mean_time, outpath=outpath, ws = 16, ol = 15, sa = 32, olsa = 28, 
            subsample = subsample , hard_subsample = hard_subsample, pertubtime = time, set_len = None)for time in time_lst)



from functions.Lucas_Kanade import *


for i in range(0, len(pertub)-time_interval): 
    print(i)
    
    u, v = lucas_kanade_np(pertub[i], pertub[i+time_interval], win=7)
    
    v= np.reshape(v,((1,v.shape[0],v.shape[1])))
    u= np.reshape(u,((1,u.shape[0],u.shape[1])))
    
    if i == 0:
        uas =  copy.copy(u)
        vas =  copy.copy(v)

    else:
        uas = np.append(uas,u,0)
        vas = np.append(vas,v,0)
    


writeNetCDF(outpath, 'WS.netcdf', 'ws', calcwindspeed(uas,vas)*0.2)

writeNetCDF(outpath, 'WD.netcdf', 'wd', calcwinddirection(uas,vas))


  
writeNetCDF(outpath, 'Tb_stab_pertub_mean.netcdf', 'Tb_pertub20Hz_means', pertub)






#
#def runATIV (tb, time_interval, mean_time, outpath, ws, ol, sa, olsa, subsample, hard_subsample, pertubtime, set_len = None, method= "greyscale", my_dpi=300):
#    
#    pertub = create_tst_pertubations_mm(tb,pertubtime)
#    
#    outpath = (outpath+"tiv/experiment_"+experiment+"_meantime_"+str(mean_time)+"_interval_"+str(time_interval)+"_method_"+method+"_WS_"+
#    str(ws)+"_OL_"+str(ol)+"_SA_"+str(sa)+"_SAOL_"+str(olsa)+"_subsample_"+str(subsample)+"_hard_subsample_"+str(hard_subsample)+"_pertubtime"+str(pertubtime)+"/")
#    
#    if not os.path.exists(outpath):
#        os.makedirs(outpath)
#            
#    print(outpath)
#
#
#
#
#    len_pertub = len(pertub)
#    if set_len is not None:
#        len_pertub = set_len
#        
#    
#    for i in range(0, len_pertub-time_interval):
#        
#        #print(i)
#        
#        
#        u, v= window_correlation_tiv(frame_a=pertub[i], frame_b=pertub[i+time_interval], window_size_x=ws, overlap_window=ol, overlap_search_area=olsa, 
#                                     search_area_size_x=sa, corr_method=method, mean_analysis = False, std_analysis = False, std_threshold = 10)
#        x, y = get_coordinates( image_size=pertub[i].shape, window_size=sa, overlap=olsa )      
#        
#        #u = remove_outliers(u,filter_size=9, sigma=1)
#        #v = remove_outliers(v, filter_size=9, sigma=1)
#        
#        
#        #plt.imshow(pertub[i], vmin = -1, vmax=1)
#        #plt.colorbar()
#        #plt.quiver(x,y,np.flipud(np.round(u,2)),np.flipud(np.round(v,2)))
#        #plt.savefig(outpath+'{:04d}'.format(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
#        #plt.close()
#        
#        
#        #plt.imshow(u)
#        
#        v= np.reshape(v,((1,v.shape[0],v.shape[1])))
#        u= np.reshape(u,((1,u.shape[0],u.shape[1])))
#    
#        if i == 0:
#            uas =  copy.copy(u)
#            vas =  copy.copy(v)
#    
#        else:
#            uas = np.append(uas,u,0)
#            vas = np.append(vas,v,0)
#    writeNetCDF(outpath, 'UAS.netcdf', 'u', uas)
#    writeNetCDF(outpath, 'VAS.netcdf', 'v', vas)
#    return()
#
#



#plt.imshow(pertub[i], vmin = -1, vmax=1)
#plt.colorbar()
#plt.quiver(x,y,np.flipud(np.round(u,2)),np.flipud(np.round(v,2)))
#plt.savefig(outpath+'{:04d}'.format(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
#plt.close()

    
    #plt.imshow(u)
    
    
#return()




