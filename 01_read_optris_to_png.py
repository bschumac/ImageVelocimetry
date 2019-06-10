#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 18:22:12 2019

@author: benjamin
"""

from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
from shutil import copyfile
from matplotlib import cm 
datapath = "/media/benjamin/Seagate Expansion Drive/T1/data/Optris_data_120119/Tier01/Flight03_O80_1616/"

# 12300
# 27000

#datapath="/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2019/Tier02/Optris_ascii/O80_220319_high_P1/"

start_img = 42000
end_img = 68000



outpath = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616_tif_viridis/"
#outpath="/media/benjamin/Seagate Expansion Drive/Darfield_Burn_Exp_Crop_2019/Tier03/Optris_ascii/O80_220319_high_P1_RGB/"

from PIL import Image
fls = os.listdir(datapath)
fls = sorted(fls, key = lambda x: x.rsplit('.', 1)[0])
# more useful:
#fls2 = sorted(fls2,key=lambda x: int(os.path.splitext(x)[0]))
my_dpi = 600





counter = 5875
for i in range(start_img,end_img, 4):
        
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
        counter +=1
