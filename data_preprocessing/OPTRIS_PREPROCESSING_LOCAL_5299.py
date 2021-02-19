#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:31:47 2020

@author: benjamin
"""


### Data needs to be available in csv files ###

# Step 1: Define experiment:

from TST_fun import *
import os
from PIL import Image
experiment = "NamTex"


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
    datapath_csv = "/mnt/Seagate_Drive1/TURF-T2/Tier01/Optris_data_csv/Flight03/"
    outpath_tb_org = "/mnt/Seagate_Drive1/TURF-T2/Tier02/Optris_data/Flight03/"
    outpath_exr_unstable = "/mnt/Seagate_Drive1/TURF-T2/Tier02/Optris_data/Flight03/Tb_exr_unstable/"
    outpath_exr_stable = "/mnt/Seagate_Drive1/TURF-T2/Tier02/Optris_data/Flight03/Tb_exr_stable/"
    rec_freq = 80
    out_freq = 2
    #T1
    start_frame = 8318
    start_img = start_frame-1200
    
    end_img = 80318
    #datapath = 

elif experiment == "Tasman_1_2":
    # Harddrive 1, 2. Start 25/02/2020 - 11:30 (UTC?)
    rec_freq = 27
    out_freq = 0.5
    
    datapath_csv = "/mnt/Seagate_Drive1/TAS1/csv_TAS1/TAS1_2/"
    outpath_tb_org = "/mnt/Seagate_Drive1/TAS1/"
    start_img = 0
    end_img = 0

elif experiment == "NamTex":
    datapath_csv = "/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/"
    outpath_tb_org = "/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/"
    outpath_exr_unstable = "/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/tb_exr_unstable/"
    outpath_exr_stable = "/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/tb_exr_stable/"
    out_freq = 9
    rec_freq = 9



# Step 2: Read files to arr



if experiment == "NamTex":
    arr = readnetcdftoarr(datapath_csv+"E01_Tb_VignStep.nc", var = 'Tb')
    
else:
    fls = os.listdir(datapath_csv)
    arr = readcsvtoarr(datapath_csv_files=datapath_csv,start_img=start_img,end_img=end_img,interval=int(rec_freq/out_freq))





# Step 3: Remove steady imagery

arr_steady = removeSteadyImages(arr, rec_feq = out_freq, print_out = True)

#arr = np.flipud(arr_steady)

writeNetCDF(outpath_tb_org, "Tb_org_"+str(out_freq)+"Hz.nc", "tb", arr)

# Step 4: arr to exr images -> Tier01

import numpy as np
import imageio


arr_steady = arr_steady.astype("float32")



for i in range(0,len(arr_steady)):
    img_rel = arr_steady[i]/np.nanmax(arr_steady[i])
    imageio.imwrite(outpath_exr_unstable+str(i+1)+'.exr', img_rel)



# Step 5: MANUAL Blender Stabilization -> Tier02
if experiment == "T1":
    saved_min = 186/10



    

# Step 6: Retrieve values from stable images
#tb_stab_lst = read_stab_red_imgs(outpath_rbg_stable)
 

arr_stab = np.empty(arr_steady.shape)

for i in range(0,len(arr_steady)):
    img = imageio.imread(outpath_exr_stable+'{:04d}'.format(i+1)+'.exr')
    img = img[:,:,0]*np.nanmax(arr_steady[i])
    arr_stab[i] = img



acc_lst = []

for i in range(0,len(arr_stab)):
    bias = np.mean(arr_stab[i][50:150,50:150]- arr_steady[i][50:150,50:150])
    acc_lst.append(bias)

print(np.mean(acc_lst))
plt.plot(acc_lst)



    

# Step 7: Write out stable tb to netcdf file -> Tier03

writeNetCDF(outpath_tb_org, "E01_Tb_stab_9Hz.nc", "Tb", arr_stab)
  
arr_stab_pertub = create_tst_pertubations_mm(arr_stab)

writeNetCDF(outpath_tb_org, "Tb_stab_pertub30s_2Hz.nc", "pertub", arr_stab_pertub)
 
arr = readnetcdftoarr("/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/E01_Tb_stab_9Hz.nc", var = 'Tb')
arr = arr[:,20:512-20,20:620]

writeNetCDF("/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/","E01_Tb_stab_9Hz.nc", "Tb", arr)



#arr = readnetcdftoarr(outpath_tb_stable+"Tb_stab_cut_red_20Hz.nc")
#
#arr = arr*10
#arr = arr +186
#arr = arr/10
#arr = np.round(arr,2)
#writeNetCDF("/mnt/Seagate_Drive1/out_test/", "Tb_stab_cut_red_20Hz_test.nc", "Tb", arr)
#  







# Former Step 4

# =============================================================================
# arr_steady = readnetcdftoarr(outpath_tb_org+"Tb_org_20Hz_test.nc", var = 'tb')
# 
# 
# 
# if np.nanmax(arr_steady) - np.nanmin(arr_steady) < 25.4:
# 
#     arr_steady = arr_steady*10
#     saved_min = (np.nanmin(arr_steady))
#     arr_steady = arr_steady-saved_min
#     arr_steady_int16 = arr_steady.astype("int16")
#     del(arr_steady)
# if redvalue:
#     
#     Tb_img = np.zeros((arr_steady_int16.shape + (3,)), dtype="uint")
#     Tb_img[:,:,:,0] = arr_steady_int16
#     
#     
#     for i in range(0,len(Tb_img)):
#         if i//100 == 0:
#             print(i)
#         write_img = Tb_img[i].astype(np.uint8)
#         im = Image.fromarray(write_img)
#         im.save(outpath_rbg_unstable+str(i+1)+".tif")
# 
#     
#     # everything from save_tb_to_red_channel
#     
# =============================================================================
