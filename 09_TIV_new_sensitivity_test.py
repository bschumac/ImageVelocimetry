#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:59:59 2019

@author: benjamin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:13:32 2019

@author: benjamin
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
import datetime
from math import sqrt
import openpiv.filters
import os
import pandas as pd

experiment = "T1"

time_a = datetime.datetime.now()


if experiment == "T0":
    datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
elif experiment == "T1":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/"
    file = h5py.File(datapath+'Tb_stab_pertub_py.nc','r')



# irgason unit calc 1 & 10 minute mean 
irg2_df = pd.read_csv('/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/IRG2/TOA5_irg_02_tiltcorr.txt', header = 0)

irg2_df.iloc[11000]
np.mean(irg2_df.iloc[:,3])
irg2_df.set_index("timestamp", inplace=True)
irg2_df.insert(0, 'row_num', range(0,len(irg2_df)))
irg2_df.loc["2019-01-12 16:33:12"]
irg2_df.head()

irg2_subset = irg2_df.iloc[231730:243996]
plt.scatter(irg2_subset.iloc[:,0],irg2_subset.iloc[:,3])
np.mean(irg2_subset.iloc[:,1])
np.mean(irg2_subset.iloc[:,2])

pertub = file.get("Tb_pertub")
pertub = np.array(pertub)
pertub = np.flip(pertub,1)

uas = []
vas = []
xas = []
yas = []


methods_lst = ["cross_corr", "greyscale", "rmse", "ssim"]

ws_lst = [8 ,36,36,36,36,24,24,36,24,16,16,16,16,24,24,36,16]
ol_lst = [7 ,25,24,27,25,23,23,35,22,15,15,15,14,23,23,35,15]
sa_lst = [16,42,42,42,36,56,48,42,36,24,40,32,24,36,56,54,20]
pos_left_corner_area = []

my_dpi = 100

methods_u_pos_lst = []
methods_3x3_u_lst = []
methods_5x5_u_lst =[]


methods_v_pos_lst = []
methods_3x3_v_lst = []
methods_5x5_v_lst =[]


lst_closest_pos_v = []
lst_3x3_v = []
lst_5x5_v =[]
lst_closest_pos_u = []
lst_3x3_u = []
lst_5x5_u =[]



#def processInput(method):
   
for method in methods_lst:
    print(method)
    lst_closest_pos_v = []
    lst_3x3_v = []
    lst_5x5_v =[]
    lst_closest_pos_u = []
    lst_3x3_u = []
    lst_5x5_u =[]
    
    for j in range(0,len(ws_lst)):#
        print(j)
        window_size=ws_lst[j]
        overlap = ol_lst[j]
        search_area_size = sa_lst[j]
        
        act_closest_pos_lst_u = []
        act_lst_3x3_lst_u = []
        act_lst_5x5_lst_u = []
    
        act_closest_pos_lst_v = []
        act_lst_3x3_lst_v = []
        act_lst_5x5_lst_v = []
        outpath = datapath+"tiv/method_"+method+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+"/"
    
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        
        if method == "greyscale" or method =="rmse" or method =="ssim":
            try:
                for i in range(139,len(pertub)-6,6):
                    #print(i)
                    frame_a = pertub[i]
                    frame_b = pertub[i+6]
                
                
                    u1, v1= window_correlation_tiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                                           search_area_size_x=search_area_size, search_area_size_y=0, corr_method=method)
                    
                    v = v1*-1
                    x1, y1 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
                    get_field_shape(image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
                    frame_a.shape[0]/get_field_shape(image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )[0]
                    frame_a.shape[1]/get_field_shape(image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )[1]
                    uas.append(u1)
                    vas.append(v1)
                    xas.append(x1)
                    yas.append(y1)
                    #x1[0][9]
                    #frame_a.shape[0]-y1[9][0]
                    u1=np.flip(u1,0)
                    v1=np.flip(v1,0)
                    
                    #plt.figure()
                    #plt.imshow(frame_a, vmin = -3, vmax=3)
                    #plt.colorbar()
                    #plt.quiver(x1,y1,u1,v1)
                    #plt.show()
                    #plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    #plt.close()
                    
                    
                    closest_pos_x = np.argmin(np.abs(x1[0]-80))+1
                    closest_pos_y = np.argmin(np.abs(np.flip(y1[:,1])-87))
                    closest_pos = (closest_pos_x,closest_pos_y)
                    u_closest_pos = u1[closest_pos]
                    v_closest_pos = v1[closest_pos]
                    act_closest_pos_lst_u.append(u_closest_pos)
                    act_closest_pos_lst_v.append(v_closest_pos)
                    
                    try:
                        closest_pos_3x3 = (closest_pos_x-1,closest_pos_y-1)
                        u_closest_pos_3x3 = u1[closest_pos_3x3[0]:closest_pos_3x3[0]+3,closest_pos_3x3[1]:closest_pos_3x3[1]+3] 
                        v_closest_pos_3x3 = v1[closest_pos_3x3[0]:closest_pos_3x3[0]+3,closest_pos_3x3[1]:closest_pos_3x3[1]+3]
                       
                       
                        
                        if np.std(v_closest_pos_3x3) > 20 or np.std(u_closest_pos_3x3) > 20:
                            raise(ValueError)
                        act_lst_3x3_lst_u.append(u_closest_pos_3x3)
                        act_lst_3x3_lst_v.append(v_closest_pos_3x3)
                    except:
                        pass
                    try:
                        closest_pos_5x5 = (closest_pos_x-2,closest_pos_y-2)
                        u_closest_pos_5x5 = u1[closest_pos_5x5[0]:closest_pos_5x5[0]+5,closest_pos_5x5[1]:closest_pos_5x5[1]+5] 
                        v_closest_pos_5x5 = v1[closest_pos_5x5[0]:closest_pos_5x5[0]+5,closest_pos_5x5[1]:closest_pos_5x5[1]+5]
                        
                        
                        if np.std(v_closest_pos_5x5) > 20 or np.std(u_closest_pos_5x5) > 20:
                            raise(ValueError)
                        act_lst_5x5_lst_u.append(u_closest_pos_5x5)
                        act_lst_5x5_lst_v.append(v_closest_pos_5x5)
                    except:
                        pass
                    
                    
                    
            except:
                pass
                
        else:
            try:
                
                for i in range(139,len(pertub)-6,6):
                    print(i)
                    frame_a = pertub[i]
                    frame_b = pertub[i+6]
                    
                    u, v = cross_correlation_aiv(frame_a, frame_b, window_size=window_size, overlap=overlap,
                                              dt=1, search_area_size=search_area_size, nfftx=None, 
                                              nffty=None,width=2, corr_method='fft', subpixel_method='gaussian')
                    
                    
                    x, y = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
                
                
                    #x=np.flip(x,0)
                    #y=np.flip(y,0)
                    u=np.flip(u,0)
                    v=np.flip(v,0)
                    #plt.imshow(y)
                    u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
                
                    
                    #plt.figure()
                    #plt.imshow(frame_a, vmin = -5, vmax=5)
                    #plt.quiver(x,y,u,v)
                    #plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    #plt.close()
                    closest_pos_x = np.argmin(np.abs(x[0]-80))+1
                    closest_pos_y = np.argmin(np.abs(np.flip(y[:,1])-87))
                    closest_pos = (closest_pos_x,closest_pos_y)
                    u_closest_pos = u[closest_pos]
                    v_closest_pos = v[closest_pos]
                    act_closest_pos_lst_u.append(u_closest_pos)
                    act_closest_pos_lst_v.append(v_closest_pos)
                    
                    try:
                        closest_pos_3x3 = (closest_pos_x-1,closest_pos_y-1)
                        u_closest_pos_3x3 = u[closest_pos_3x3[0]:closest_pos_3x3[0]+3,closest_pos_3x3[1]:closest_pos_3x3[1]+3] 
                        v_closest_pos_3x3 = v[closest_pos_3x3[0]:closest_pos_3x3[0]+3,closest_pos_3x3[1]:closest_pos_3x3[1]+3]
                       
                       
                        
                        if np.std(v_closest_pos_3x3) > 20 or np.std(u_closest_pos_3x3) > 20:
                            raise(ValueError)
                        act_lst_3x3_lst_u.append(u_closest_pos_3x3)
                        act_lst_3x3_lst_v.append(v_closest_pos_3x3)
                    except:
                        pass
                    try:
                        closest_pos_5x5 = (closest_pos_x-2,closest_pos_y-2)
                        u_closest_pos_5x5 = u[closest_pos_5x5[0]:closest_pos_5x5[0]+5,closest_pos_5x5[1]:closest_pos_5x5[1]+5] 
                        v_closest_pos_5x5 = v[closest_pos_5x5[0]:closest_pos_5x5[0]+5,closest_pos_5x5[1]:closest_pos_5x5[1]+5]
                        
                        
                        if np.std(v_closest_pos_5x5) > 20 or np.std(u_closest_pos_5x5) > 20:
                            raise(ValueError)
                        act_lst_5x5_lst_u.append(u_closest_pos_5x5)
                        act_lst_5x5_lst_v.append(v_closest_pos_5x5)
                    except:
                        pass

            except:
                pass
            
        lst_closest_pos_u.append(act_closest_pos_lst_u)
        lst_3x3_u.append(act_lst_3x3_lst_u)
        lst_5x5_u.append(act_lst_5x5_lst_u)
        
        lst_closest_pos_v.append(act_closest_pos_lst_v)
        lst_3x3_v.append(act_lst_3x3_lst_v)
        lst_5x5_v.append(act_lst_5x5_lst_v)
    
    methods_u_pos_lst.append(lst_closest_pos_u)
    methods_3x3_u_lst.append(lst_3x3_u)
    methods_5x5_u_lst.append(lst_5x5_u)
    
    
    methods_v_pos_lst.append(lst_closest_pos_v)
    methods_3x3_v_lst.append(lst_3x3_v)
    methods_5x5_v_lst.append(lst_5x5_v)
        
    #return(lst_closest_pos_u,lst_3x3_u,lst_5x5_u,lst_closest_pos_v,lst_3x3_v,lst_5x5_v)


for m in range(0,len(ws_lst)):
    print("u mean:")
    print(np.mean(methods_u_pos_lst[0][m]))
    
    print("v mean:")
    print(np.mean(methods_v_pos_lst[0][m]))




# cc dataframe build
 
cc_IV_array = np.array([ws_lst,ol_lst,sa_lst])     
cc_IV_array = np.flipud(np.rot90(cc_IV_array))    
    
u_minmean = np.zeros((17,10))
v_minmean = np.zeros((17,10))
    
def minmean(lst):
    retlst=[]
    for i in range(0,len(lst)-60,60):
        retlst.append(np.mean(lst[i:i+60]))
    return retlst

def calc_mean(lst):
    retlst = []
    for i in range(0,len(lst)):
        retlst.append(np.mean(lst[i]))
    return retlst
    

for m in range(0,len(ws_lst)):
    u_minmean[m,] = minmean(methods_u_pos_lst[0][m])
    v_minmean[m,] = minmean(methods_v_pos_lst[0][m])


u_mean = np.reshape(np.array(calc_mean(methods_u_pos_lst[0])),(17,1))
v_mean = np.reshape(np.array(calc_mean(methods_v_pos_lst[0])),(17,1))


   
u_3x3_minmean = np.zeros((17,10))
v_3x3_minmean = np.zeros((17,10))
    
   

for m in range(0,len(ws_lst)):
    u_3x3_minmean[m,] = minmean(methods_3x3_u_lst[0][m])
    v_3x3_minmean[m,] = minmean(methods_3x3_v_lst[0][m])


u_3x3_mean = np.reshape(np.array(calc_mean(methods_3x3_u_lst[0])),(17,1))
v_3x3_mean = np.reshape(np.array(calc_mean(methods_3x3_v_lst[0])),(17,1))




   
u_5x5_minmean = np.zeros((17,10))
v_5x5_minmean = np.zeros((17,10))
    
   

for m in range(0,len(ws_lst)):
    u_5x5_minmean[m,] = minmean(methods_5x5_u_lst[0][m])
    v_5x5_minmean[m,] = minmean(methods_5x5_v_lst[0][m])


u_5x5_mean = np.reshape(np.array(calc_mean(methods_5x5_u_lst[0])),(17,1))
v_5x5_mean = np.reshape(np.array(calc_mean(methods_5x5_v_lst[0])),(17,1))












cc_IV_array= np.concatenate((cc_IV_array, u_minmean, v_minmean, u_mean, v_mean, u_3x3_minmean, v_3x3_minmean, u_3x3_mean, v_3x3_mean  ), axis=1)



cc_d = {'ws': ws_lst, 'ol': ol_lst, 'sa' : sa_lst}
cc_df = pd.DataFrame(data=cc_d)

cc_df.append(u_minmean)



            
#num_cores = 4
    
#u, u3x3, u5x5, v, v3x3, v5x5 = Parallel(n_jobs=num_cores)(delayed(processInput)(method) for method in methods_lst)



