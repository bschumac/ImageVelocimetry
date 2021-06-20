#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:49:48 2020

@author: benjamin
"""

import numpy as np
from functions.TST_fun import readnetcdftoarr, writeNetCDF
from PIL import Image
import matplotlib.pyplot as plt

datapath = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier02/Optris_data/"
outpath = datapath+"Flight01_tif_virdris_27Hz_cut/"



Tb_org = readnetcdftoarr(datapath+"Tb_org_cut_27Hz.nc")

 
Tb_org[Tb_org <15] = 10
Tb_org = Tb_org-10
Tb_org = Tb_org*10

my_dpi = 300
fig = plt.figure(figsize=(382/my_dpi, 288/my_dpi), dpi=my_dpi)
ax1 = plt.subplot(111)
#u_xz_norm =  (u_xz - Tb_MIN) / (Tb_MAX-Tb_MIN)
im=ax1.imshow(Tb_org[255],interpolation=None,cmap="Greys")
ax1.axis("off")
im.axes.get_xaxis().set_visible(False)
im.axes.get_yaxis().set_visible(False)
plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.savefig(datapath+"test_img"+".tif",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
plt.close()

read_img = np.array(plt.imread(datapath+"/test_img.tif")[:,:,:3])
read_img[:,:,1:3] = 0


for i in range(0,len(Tb_org)):
    if i//100 == 0:
        print(i)
    read_img[:,:,0] = Tb_org[i]
    im = Image.fromarray(read_img)
    im.save(outpath+str(i+1)+".tif")

    


Tb_org = readnetcdftoarr(datapath+"Tb_org_cut_27Hz.nc")
tb_stab_lst = read_stab_red_imgs("/home/benjamin/Met_ParametersTST/Pre_Fire/Tier02/Optris_data/Flight01_tif_red_27Hz_cut_stab/", Tb_org)

outpath2 = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"
writeNetCDF(outpath2, "Tb_stab_cut_red_27Hz", "Tb", tb_stab_lst[0])









Tb_org = Tb_org.astype('int')


for i in range(0,len(Tb_org)):
    
    