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
experiment = "T1"


if experiment == "T1":
    datapath_csv = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616/"
    rec_freq = 80
    out_freq = 20
    #T1
    start_img = 18500
    end_img = 68000
    redvalue = True

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

arr = removeSteadyImages(arr, rec_feq = int(rec_freq/out_freq), print_out = True)

# Step 4: arr to RGB images -> Tier01

if redvalue:
    
    # everything from save_tb_to_red_channel
    
# Step 5: MANUAL Blender Stabilization -> Tier02
    

# Step 6: Retrieve values from stable images
    

# Step 7: Write out stable tb to netcdf file -> Tier03






