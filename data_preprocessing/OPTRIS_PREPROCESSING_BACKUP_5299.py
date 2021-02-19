#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:31:47 2020

@author: benjamin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:31:47 2020

@author: benjamin
"""


### Data needs to be available in csv files ###

# Step 1: Define experiment:
import sys
#sys.path.insert(1, '/home/rccuser/.jupyter/code/ImageVelocimetry/functions/')

from TST_fun import *
import os
from PIL import Image
<<<<<<< HEAD
experiment = "NamTex"

=======
experiment = "namtex"
import numpy as np
import imageio
>>>>>>> 6b257267229fefc2b1cdf988f5974f5e62a4b382

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
    out_freq = 20
    datapath_csv = "/home/bsc/Met_ParametersTST/T2/Tier01/Flight03/"
    outpath_tb_org = "/home/bsc/Met_ParametersTST/T2/Tier01/"
    outpath_exr_unstable = "/home/bsc/Met_ParametersTST/T2/Tier01/Tb_exr_"+str(out_freq)+"_unstable/"
    outpath_exr_stable = "/home/bsc/Met_ParametersTST/T2/Tier01/Tb_exr_"+str(out_freq)+"Hz_stable/"
    
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

<<<<<<< HEAD
elif experiment == "NamTex":
    datapath_csv = "/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/"
    outpath_tb_org = "/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/"
    outpath_exr_unstable = "/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/tb_exr_unstable/"
    outpath_exr_stable = "/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/tb_exr_stable/"
    out_freq = 9
    rec_freq = 9
=======
    
elif experiment == "namtex":
   
    rec_freq = 8
    out_freq = 8

    experiment = "Experiment10"
    #datapath_csv = "/data/Met_ParametersTST/NamTEX//Experiment02/Level1/Export2/"
    outpath_tb_org = "/home/benjamin/Met_ParametersTST/NamTEX/"+experiment+"/Level01/"
    outpath_exr_unstable = "//home/benjamin/Met_ParametersTST/NamTEX/"+experiment+"/Level01/Tb_exr"+str(out_freq)+"Hz_unstable/"
    outpath_exr_stable = "/home/benjamin/Met_ParametersTST/NamTEX/"+experiment+"/Level01/Tb_exr"+str(out_freq)+"Hz_stable/"
    start_img = 0
    end_img = 0

elif experiment == "namtex2":
   
    rec_freq = 1
    out_freq = 1
    
    datapath_csv = "/data/Met_ParametersTST/NamTEX/Tier01/Export2/"
    outpath_tb_org = "/data/Met_ParametersTST/NamTEX/Tier01/"
    outpath_exr_unstable = "/data/Met_ParametersTST/NamTEX/Tier01/Tb_exr"+str(out_freq)+"Hz_unstable/"
    outpath_exr_stable = "/data/Met_ParametersTST/NamTEX/Tier01/Tb_exr"+str(out_freq)+"Hz_stable/"
    start_img = 0
    end_img = 0

elif experiment == "fire_2019_p1":
    rec_freq = 80
    datapath_csv = "/data/FIRE/Darfield_2019/Tier01/O80_220319_high_P1/"
    outpath_tb_org = "/data/FIRE/Darfield_2019/Tier01/"

>>>>>>> 6b257267229fefc2b1cdf988f5974f5e62a4b382


#arr = readnetcdftoarr(outpath_tb_stable+"Tb_stab_cut_red_20Hz.nc")
#
#arr = arr*10
#arr = arr +186
#arr = arr/10
#arr = np.round(arr,2)
#writeNetCDF("/mnt/Seagate_Drive1/out_test/", "Tb_stab_cut_red_20Hz_test.nc", "Tb", arr)
#  

# Step 2: Read files to arr



<<<<<<< HEAD
if experiment == "NamTex":
    arr = readnetcdftoarr(datapath_csv+"E01_Tb_VignStep.nc", var = 'Tb')
    
else:
    fls = os.listdir(datapath_csv)
    arr = readcsvtoarr(datapath_csv_files=datapath_csv,start_img=start_img,end_img=end_img,interval=int(rec_freq/out_freq))
=======


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

# Step 2: Read files to arr

fls = os.listdir(datapath_csv)
arr = readcsvtoarr(datapath_csv_files=datapath_csv,start_img=start_img,end_img=end_img,interval=int(rec_freq/out_freq))
>>>>>>> 6b257267229fefc2b1cdf988f5974f5e62a4b382


#arr = readnetcdftoarr(outpath_tb_org+"E10_Tb_VignStep.nc", var = 'Tb')



#outpath_tb_org

# Step 3: Remove steady imagery

arr_steady = removeSteadyImages(arr, rec_feq = out_freq, print_out = True)

#arr = np.flipud(arr_steady)

#writeNetCDF(outpath_tb_org, "Tb_org_"+str(out_freq)+"Hz.nc", "tb", arr)
#arr = readnetcdftoarr("/data/Met_ParametersTST/NamTEX/Tier01//Tb_org_9Hz.nc", var = 'tb')

import progressbar

if not os.path.exists(outpath_exr_unstable):
    os.makedirs(outpath_exr_unstable)
  
if not os.path.exists(outpath_exr_stable):
    os.makedirs(outpath_exr_stable)



# Step 4: arr to exr images -> Tier01



arr_steady = arr_steady.astype("float32")


<<<<<<< HEAD

for i in range(0,len(arr_steady)):
    img_rel = arr_steady[i]/np.nanmax(arr_steady[i])
    imageio.imwrite(outpath_exr_unstable+str(i+1)+'.exr', img_rel)
=======
bar = progressbar.ProgressBar(maxval=arr_steady.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) 
bar.start()
bar_iterator = 0
for i in range(0,len(arr_steady)):
    img_rel = arr_steady[i]/np.nanmax(arr_steady[i])
    imageio.imwrite(outpath_exr_unstable+str(i+1)+'.exr', img_rel)
    bar.update(bar_iterator+1)
    bar_iterator +=1
bar.finish()   


>>>>>>> 6b257267229fefc2b1cdf988f5974f5e62a4b382



# Step 5: MANUAL Blender Stabilization -> Tier02
if experiment == "T1":
    saved_min = 186/10



del(arr)

# Step 6: Retrieve values from stable images
#tb_stab_lst = read_stab_red_imgs(outpath_rbg_stable)
 

arr_stab = np.empty(arr_steady.shape)

for i in range(0,len(arr_steady)):
    if i%100 == 0:
        print(i)
    img = imageio.imread(outpath_exr_stable+'{:04d}'.format(i+1)+'.exr')
    img = img[:,:,0]*np.nanmax(arr_steady[i])
    arr_stab[i] = img



acc_lst = []

for i in range(0,len(arr_stab)):
    bias = np.mean(arr_stab[i][50:150,50:150]- arr_steady[i][50:150,50:150])
    acc_lst.append(bias)

print(np.mean(acc_lst))
plt.plot(acc_lst)

del(arr_steady)



# Step 7: Write out stable tb to netcdf file -> Tier03
arr_stab = arr_stab[:, 50:512-50,50:640-50]

np.save(outpath_tb_org+"E010_Tb_stab_8Hz.npy", arr_stab)    

arr_stab = np.load(outpath_tb_org+"E010_Tb_stab_8Hz.npy")

<<<<<<< HEAD
writeNetCDF(outpath_tb_org, "E01_Tb_stab_9Hz.nc", "Tb", arr_stab)
  
arr_stab_pertub = create_tst_pertubations_mm(arr_stab)

writeNetCDF(outpath_tb_org, "Tb_stab_pertub30s_2Hz.nc", "pertub", arr_stab_pertub)
 
arr = readnetcdftoarr("/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/E01_Tb_stab_9Hz.nc", var = 'Tb')
arr = arr[:,20:512-20,20:620]

writeNetCDF("/home/benjamin/Met_ParametersTST/NamTex/Experiment01/Level01/","E01_Tb_stab_9Hz.nc", "Tb", arr)
=======
writeNetCDF(outpath_tb_org, "E10_Tb_stab_8Hz.nc", "Tb", arr_stab)
  
arr_stab_pertub = create_tst_pertubations_mm(arr_stab)

>>>>>>> 6b257267229fefc2b1cdf988f5974f5e62a4b382



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
