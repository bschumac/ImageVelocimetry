#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 21:23:57 2019

@author: benjamin
"""

  
import numpy as np
import os
#fp = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/RMSE_lst.txt"
#
#
#
#
#x = np.loadtxt(fp, delimiter='\n')
#np.nanmean(x)
#np.nanstd(x)
#
#
#
#
#
#
#datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
#
#file = h5py.File(datapath+'Tb_stab_pertub20s_py_virdris_20Hz.nc','r')
#tb= file.get("Tb_pertub")
#
#pertub = np.array(tb)
#
#std_pertub= np.nanstd(pertub,0)
#
#
#fig = plt.figure()
#plt.imshow(std_pertub)
#plt.colorbar()
#plt.savefig(datapath+str("std_Tb_stab_pertub20s_py_virdris_20Hz")+".png",dpi=600,bbox_inches='tight',pad_inches = 0,transparent=False)
#plt.close()

org_datapath = "/data/TST/data/"
img_datapath1 = "/data/TST/data/Tier01/Flight03_O80_1616_tif_viridis/"
img_datapath2="/data/TST/data/Tier02/Flight03_O80_1616_stab_tif_virdris_20Hz/"
datapath_csv_files = "/data/TST/data/Tier01/Flight03_O80_1616/"
fls = os.listdir(datapath_csv_files)
fls = sorted(fls, key = lambda x: x.rsplit('.', 1)[0])
    
    

print("Reading original data to RAM...")
counter = 0
for i in range(start_img,end_img, interval): 
    
    if counter%100 == 0:
        print(str(counter)+" of "+str((end_img-start_img)/4))
    my_data = np.genfromtxt(datapath_csv_files+fls[i], delimiter=',', skip_header=1)
    my_data = np.reshape(my_data,(1,my_data.shape[0],my_data.shape[1]))
    if counter == 0:
        org_data = copy.copy(my_data)
    else:
        org_data = np.append(org_data,my_data,0)
    #org_data[counter] = my_data 
    counter+=1
print("...finished!")

writeNetCDF(org_datapath,"Tb_org_20Hz.nc","Tb",org_data)