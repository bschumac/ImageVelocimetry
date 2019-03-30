#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:13:32 2019

@author: benjamin
DEPRICATED - too slow

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

datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
outpath = datapath+"test_tiv/"

file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
pertub = file.get("Tb_pertub")
pertub = np.array(pertub)
pertub = np.flip(pertub,1)
inter_window_size = 9
inter_window_size = int((inter_window_size-1)/2) 
max_displacement = 25
num_cores = multiprocessing.cpu_count()-20
begin_value_images = 387
end_value_images = 391
end_of_u = 300
end_of_v = 288
begin_of_v = 20
begin_of_u = 0

pertub= pertub[begin_value_images:end_value_images]
pertub = pertub[:,begin_of_v:end_of_v,begin_of_u:end_of_u]




pertub_cut = pertub[:,(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]
pertub_cut.shape

#writeNetCDF(outpath,"Tb_pertub_cut_"+str(begin_value_images)+"-"+str(end_value_images)+".nc", "Tb_pertub",pertub_cut)


iterx = max_displacement+inter_window_size -1
itery = max_displacement+inter_window_size -1


def processInput(n, pertub, iterx, itery, maxval, ttest= False):
    n = 0
    u = np.zeros((pertub.shape[1],pertub.shape[2]))
    v = np.zeros((pertub.shape[1],pertub.shape[2]))
    if ttest:
        u_ttest = np.zeros((pertub.shape[1],pertub.shape[2]))
        v_ttest = np.zeros((pertub.shape[1],pertub.shape[2]))
    print(n)
    
    for l in range(max_displacement+inter_window_size,(pertub.shape[1])-(max_displacement+inter_window_size)):
        
        print(l)
        print("of "+str((pertub.shape[1])-(max_displacement+inter_window_size)))
        
        centery = l
        iterx+=1
        itery = max_displacement+inter_window_size -1
        
        for m in range(max_displacement+inter_window_size,(pertub.shape[2])-(max_displacement+inter_window_size)):
            centerx = m
            itery+=1
            inter_wind1 = pertub[n][centery-inter_window_size:centery+inter_window_size,centerx-inter_window_size:centerx+inter_window_size]
            
            # displace interigatnp.save(datapath+"v_TIV_test", )ion window 2 by various 
            acc_dif_lst = []
            acc_iter_lst = []
            if ttest:
                acc_ttest_p_val_lst = []
            for i in range(-max_displacement,max_displacement):
                disp_centerx = centerx+i
                for k in range(-max_displacement,max_displacement):
                        disp_centery = centery+k
                        inter_wind2 = pertub[n+1][disp_centery-inter_window_size:disp_centery+inter_window_size,disp_centerx-inter_window_size:disp_centerx+inter_window_size]
                        dif = inter_wind1 - inter_wind2
                        acc_dif = sum(sum(abs(dif)))
                        acc_dif_lst.append(acc_dif)
                        acc_iter_lst.append((i,k))
                        if ttest:
                            t2, p2 = stats.ttest_ind(inter_wind1.flatten(),inter_wind2.flatten(),equal_var=False)
                            acc_ttest_p_val_lst.append(p2)
                        
            
            acc_dif_lst[acc_dif_lst.index(min(acc_dif_lst))]
            
            #print(acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))])
            v[iterx,itery] = acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))][0]
            u[iterx,itery] = acc_iter_lst[acc_dif_lst.index(min(acc_dif_lst))][1]
            
            #u_ttest[iterx,itery] = acc_iter_lst[acc_ttest_p_val_lst.index(max(acc_ttest_p_val_lst))][0]
            #v_ttest[iterx,itery] = acc_iter_lst[acc_ttest_p_val_lst.index(max(acc_ttest_p_val_lst))][1]
            
    if ttest:
        return(np.stack((u, v, u_ttest, v_ttest)))
    else:
        return(np.stack((u, v)))
        
        #plt.imshow(u,cmap = cm.jet, vmin=np.min(u), vmax=np.max(u))
        #plt.colorbar()
 
    


out_u_v = Parallel(n_jobs=num_cores)(delayed(processInput)(n, pertub, iterx = max_displacement+inter_window_size -1,itery = max_displacement+inter_window_size -1 , maxval = 198, ttest= True) for n in range(begin_value_images-begin_value_images,end_value_images-begin_value_images-1))

 

z = 0
x = 0


output_file_u = out_u_v[z][x,(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]

x = 2
output_file_u_ttest = out_u_v[z][x,(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]


x = 1
output_file_v = out_u_v[z][x,(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]

x = 3
output_file_v_ttest = out_u_v[z][x,(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]



writeNetCDF(outpath,"u_TIV_test.nc", "u",output_file_u)

writeNetCDF(outpath,"v_TIV_test.nc", "v",output_file_v)

writeNetCDF(outpath,"u_TIV_test_ttest.nc", "u",output_file_u_ttest)

writeNetCDF(outpath,"v_TIV_test_ttest.nc", "v",output_file_v_ttest)


# Analysis: Testing...


plt.imshow(output_file_v_ttest,cmap = cm.jet, vmin=np.min(output_file_v_ttest), vmax=np.max(output_file_v_ttest))
plt.colorbar()
plt.imshow(output_file_v,cmap = cm.jet, vmin=np.min(output_file_v), vmax=np.max(output_file_v))

#plt.imshow(v,cmap = cm.jet, vmin=np.min(v), vmax=np.max(v))
#plt.colorbar()

len(test[z][x].shape)

max_displacement+inter_window_size





plt.imshow(test[z][x][(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)], cmap = cm.jet)

plt.imshow(test[z][x], cmap = cm.jet)

array_size_y = end_of_v-begin_of_v-(max_displacement+inter_window_size)-(max_displacement+inter_window_size)
array_size_x = end_of_u-begin_of_u-(max_displacement+inter_window_size)-(max_displacement+inter_window_size)

import math as m
erg_arr_direc = np.empty((len(test),array_size_y,array_size_x))
erg_arr_len = np.empty((len(test),array_size_y,array_size_x))



for x in range(0,len(test)):
    print(x)
    act_mat_uas = np.matrix(test[x][0])
    act_mat_vas = np.matrix(test[x][1])

    act_mat_uas = act_mat_uas[(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]
    act_mat_vas = act_mat_vas[(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]
    act_mat_uas[np.isnan(act_mat_uas)] = 60
    act_mat_vas[np.isnan(act_mat_vas)] = -60
    erg_vlen_abs = np.sqrt(np.square(act_mat_uas)+np.square(act_mat_vas))
    erg_dir_rad = np.arctan2(act_mat_uas/erg_vlen_abs, act_mat_vas/erg_vlen_abs)
    erg_dir_deg = (erg_dir_rad * 180)/m.pi
    
    erg_dir_deg_pos = np.where(erg_dir_deg < 0.0, erg_dir_deg+360, erg_dir_deg)
    
    erg_dir_deg_pos[erg_vlen_abs>50] = np.nan
    erg_vlen_abs[erg_vlen_abs>50] = np.nan
    
    erg_arr_direc[x] = erg_dir_deg_pos#(np.rot90(erg_dir_deg_pos,1))
    erg_arr_len[x] = erg_vlen_abs#(np.rot90(erg_vlen_abs,1))
    #erg_arr_direc[x] = erg_arr_direc#(np.fliplr(erg_arr_direc[x]))
    #erg_arr_len[x] = erg_arr_len#(np.fliplr(erg_arr_len[x]))


erg_arr_direc[1].shape

plt.imshow(act_mat_uas, cmap = cm.jet)
plt.imshow(act_mat_vas, cmap = cm.jet)

plt.imshow(erg_vlen_abs, cmap = cm.jet)
plt.imshow(erg_dir_deg_pos, cmap = cm.jet)
plt.colorbar()




a = erg_dir_deg_pos
a.shape
a[np.isnan(a)] = 90
a.shape
a[ (a < 225) & (a > 45) ] = np.nan
my_dpi = 100
for i in range(0,5):
    plt.imshow(test[i][0], cmap = cm.jet)
    if i == 0:
        plt.colorbar()
    plt.savefig(datapath+"test_tiv/uas_2"+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)