#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:31:47 2020

@author: benjamin
"""


### Data needs to be available in csv files ###

# Step 1: Define experiment:

from functions.TST_fun import *
import os
from PIL import Image
experiment = "T1"


if experiment == "T1":
    redvalue = True
    datapath_csv = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616/"
    outpath_tb_org = "/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/"
    outpath_rbg_unstable = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616_tif_redvalue/"
    outpath_rbg_stable = "/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Flight03_O80_1616_stab_tif_redvalue/"
    outpath_tb_stable = "/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/"
    
    
    rec_freq = 80
    out_freq = 20
    #T1
    start_img = 18500
    end_img = 68000
    

elif experiment == "pre_fire_":
    rec_freq = 27
    datapath_csv = "/home/benjamin/Met_ParametersTST/Pre_Fire/Tier03/Optris_data/"
elif experiment == "T2":
    rec_freq = 80
    #datapath = 
   


# Step 2: Read files to arr


fls = os.listdir(datapath_csv)
len(fls)
arr = readcsvtoarr(datapath_csv_files=datapath_csv,start_img=start_img,end_img=end_img,interval=int(rec_freq/out_freq))


# Step 3: Remove steady imagery

arr_steady = removeSteadyImages(arr, rec_feq = 20, print_out = True)

writeNetCDF(outpath_tb_org, "Tb_org_20Hz_test.nc", "tb", arr_steady)

# Step 4: arr to RGB images -> Tier01

arr_steady = readnetcdftoarr(outpath_tb_org+"Tb_org_20Hz_test.nc", var = 'tb')



if np.nanmax(arr_steady) - np.nanmin(arr_steady) < 25.4:

    arr_steady = arr_steady*10
    saved_min = (np.nanmin(arr_steady))
    arr_steady = arr_steady-saved_min
    arr_steady_int16 = arr_steady.astype("int16")
    del(arr_steady)
if redvalue:
    
    Tb_img = np.zeros((arr_steady_int16.shape + (3,)), dtype="uint")
    Tb_img[:,:,:,0] = arr_steady_int16
    
    
    for i in range(0,len(Tb_img)):
        if i//100 == 0:
            print(i)
        write_img = Tb_img[i].astype(np.uint8)
        im = Image.fromarray(write_img)
        im.save(outpath_rbg_unstable+str(i+1)+".tif")

    
    # everything from save_tb_to_red_channel
    
# Step 5: MANUAL Blender Stabilization -> Tier02
if experiment == "T1":
    saved_min = 186/10



    

# Step 6: Retrieve values from stable images
tb_stab_lst = read_stab_red_imgs(outpath_rbg_stable)
 

writeNetCDF(outpath_tb_stable, "Tb_stab_cut_red_20Hz.nc", "Tb", tb_stab_lst[0])
  

    

# Step 7: Write out stable tb to netcdf file -> Tier03


arr = readnetcdftoarr(outpath_tb_stable+"Tb_stab_cut_red_20Hz.nc")

arr = arr+18.6
arr = np.round(arr,2)
writeNetCDF(outpath_tb_stable, "Tb_stab_cut_red_20Hz_test.nc", "Tb", arr)
  



