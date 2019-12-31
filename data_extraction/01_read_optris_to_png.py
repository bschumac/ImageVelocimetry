#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 18:22:12 2019

@author: benjamin
"""

from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from matplotlib import cm 
from functions.TST_fun import writeNetCDF

#datapath = "/media/benjamin/Seagate Expansion Drive/T1/data/Optris_data_120119/Tier01/Flight03_O80_1616/"

#T0?
# 12300
# 27000

#T1
#18500
#68000

datapath="/home/benjamin/Met_ParametersTST/Pre_Fire/Tier01/Optris_ascii/"

start_img = 10000
end_img = 15500



outpath = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier02/Optris_data/Flight01_tif_virdris_1Hz/"
outpath_netcdf = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier02/Optris_data/"

if not os.path.exists(outpath):
    os.makedirs(outpath)
#outpath="/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2019/Tier03/Optris_ascii/O80_220319_high_P1_RGB/"


fls = os.listdir(datapath)
fls = sorted(fls, key = lambda x: x.rsplit('.', 1)[0])
# more useful:
#fls2 = sorted(fls2,key=lambda x: int(os.path.splitext(x)[0]))
my_dpi = 600


counter = 1
for i in range(start_img,end_img, 27):
        my_data = genfromtxt(datapath+fls[i], delimiter=',', skip_header=1)
        print("Writing File " +str(i)+".tif from "+str(len(fls)))
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

writeNetCDF(outpath_netcdf,"Tb_org_1Hz.nc","Tb",final_tb)