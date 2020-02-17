#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:54:32 2020

@author: benjamin
"""



import numpy as np
import matplotlib.pyplot as plt
from functions.TST_fun import readnetcdftoarr, writeNetCDF
#import openpiv.filters

import copy


arr = readnetcdftoarr("/home/benjamin/Met_ParametersTST/Pre_Fire/Tier02/Optris_data/Tb_org_27Hz.nc")




outpath_netcdf = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier02/Optris_data/"
writeNetCDF(outpath_netcdf,"Tb_org_cut_27Hz.nc","Tb",arr)


counter = 1
outpath = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier02/Optris_data/Flight01_tif_virdris_27Hz_cut/"
my_dpi = 600
for i in range(0,len(arr), 1):
        my_data = arr[i]
        #genfromtxt(datapath+fls[i], delimiter=',', skip_header=1)
        print("Writing File " +str(i)+".tif from "+str(len(arr)))
        fig = plt.figure(figsize=(382/my_dpi, 288/my_dpi), dpi=my_dpi)
        ax1 = plt.subplot(111)
        #u_xz_norm =  (u_xz - Tb_MIN) / (Tb_MAX-Tb_MIN)
        im=ax1.imshow(my_data,interpolation=None,cmap=cm.viridis)
        ax1.axis("off")
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
        plt.savefig(outpath+str(counter)+".tif",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
        plt.close()
        my_data= np.reshape(my_data,((1,my_data.shape[0],my_data.shape[1])))
        if counter == 1:
            final_tb = copy.copy(my_data)
        else:
            final_tb = np.append(final_tb,my_data,0)
        counter +=1