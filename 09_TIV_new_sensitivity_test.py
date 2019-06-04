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


def minmean(lst, min_offset = 60):
    retlst=[]
    for i in range(0,len(lst)-min_offset,min_offset):
        retlst.append(np.nanmean(lst[i:i+min_offset]))
    return retlst   

def rmse(m, o):
    if len(m.shape) == 3: 
        return np.sqrt(np.mean((m-o)**2,(1,2)))
    if len(m.shape) == 1:
        return np.sqrt(np.mean((m-o)**2))
    




experiment = "T1"

time_a = datetime.datetime.now()


if experiment == "T0":
    datapath = "/home/benjamin/Met_ParametersTST/T0/data/"
    file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
elif experiment == "T1":
    datapath = "/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"
    file = h5py.File(datapath+'Tb_stab_pertub_py.nc','r')



# irgason unit calc 1 & 10 minute mean 
irg2_df = pd.read_csv('/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/IRG2/TOA5_irg_02_tiltcorr.txt', header = 0)

irg2_df.iloc[11000]
np.mean(irg2_df.iloc[:,3])
irg2_df.set_index("timestamp", inplace=True)
irg2_df.insert(0, 'row_num', range(0,len(irg2_df)))
irg2_df.loc["2019-01-12 16:22:54"]
irg2_df.loc["2019-01-12 16:33:13"]
irg2_df.head()
irg2_subset = irg2_df.iloc[231730:244016]


irg1_df = pd.read_csv('/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/IRG1/TOA5_irg_01_tiltcorr.txt', header = 0)

irg1_df.iloc[11000]
np.mean(irg2_df.iloc[:,3])
irg1_df.set_index("timestamp", inplace=True)
irg1_df.insert(0, 'row_num', range(0,len(irg1_df)))
irg1_df.loc["2019-01-12 16:21:40.7"]
irg1_df.loc["2019-01-12 16:31:58.7"]

irg1_subset = irg2_df.iloc[231471:243831]




plt.scatter(irg2_subset.iloc[:,0],irg2_subset.iloc[:,3])




pertub = file.get("Tb_pertub")
pertub = np.array(pertub)
pertub = np.flip(pertub,1)


Z0 = 0.01
H2 = 0.015

wd_mean_irg2 = calcwinddirection(np.mean(irg2_subset.iloc[:,1]), np.mean(irg2_subset.iloc[:,2]))
wd_mean_irg1 = calcwinddirection(np.mean(irg1_subset.iloc[:,1]), np.mean(irg1_subset.iloc[:,2]))
ws_mean_irg1 = calcwindspeed(np.mean(irg1_subset.iloc[:,1]), np.mean(irg1_subset.iloc[:,2]))

ws_irg2 =  calcwindspeed(irg2_subset.iloc[:,1], irg2_subset.iloc[:,2])
ws_mean_irg2 = calcwindspeed(np.mean(irg2_subset.iloc[:,1]), np.mean(irg2_subset.iloc[:,2]))
ws_log_mean_irg2=  ws_mean_irg2*(log(H2/Z0)/log(1.5/Z0))


ws_log_irg2 = ws_irg2*(log(H2/Z0)/log(1.5/Z0))

ws_minmean_irg2 = minmean(ws_irg2, 1200) 

ws_log_minmean_irg2 = minmean(ws_log_irg2, 1200) 



wd_irg2 = calcwinddirection(irg2_subset.iloc[:,1], irg2_subset.iloc[:,2])


u_minmean_irg2 = u_minmean_irg2 = minmean(irg2_subset.iloc[:,1], 1200) 

v_minmean_irg2 = minmean(irg2_subset.iloc[:,2], 1200) 

wd_minmean_irg2 = calcwinddirection(u_minmean_irg2, v_minmean_irg2)



#,
methods_lst = [ "cross_corr", "greyscale", "rmse", "ssim"]

ws_lst = [8 ,36,36,36,36,24,24,36,24,16,16,16,16,24,36,16]
ol_lst = [7 ,25,24,27,25,23,23,35,22,15,15,15,14,23,35,15]
sa_lst = [16,42,42,42,36,56,48,42,36,24,40,32,24,36,54,20]





res_df = pd.DataFrame({"method":["IRG2"], 
                         "ws":[0],
                         "ol":[0],
                         "sa":[0],
                         "pos_y":[0],
                         "pos_x":[0],
                         "rmse":[0],
                         "mean_wd": [wd_mean_irg2],
                         "mean_ws": [ws_log_mean_irg2],
                         "diff_direc": [0],
                         "diff_ws": [0]}) 

minmean_res_df = pd.DataFrame({"method":["IRG2", "IRG2_log", "IRG2_wd"], 
                         "ws1":[ws_minmean_irg2[0],ws_log_minmean_irg2[0],wd_minmean_irg2[0]],
                         "ws2":[ws_minmean_irg2[1],ws_log_minmean_irg2[1],wd_minmean_irg2[1]],
                         "ws3":[ws_minmean_irg2[2],ws_log_minmean_irg2[2],wd_minmean_irg2[2]],
                         "ws4":[ws_minmean_irg2[3],ws_log_minmean_irg2[3],wd_minmean_irg2[3]],
                         "ws5":[ws_minmean_irg2[4],ws_log_minmean_irg2[4],wd_minmean_irg2[4]],
                         "ws6":[ws_minmean_irg2[5],ws_log_minmean_irg2[5],wd_minmean_irg2[5]],
                         "ws7":[ws_minmean_irg2[6],ws_log_minmean_irg2[6],wd_minmean_irg2[6]],           
                         "ws8":[ws_minmean_irg2[7],ws_log_minmean_irg2[7],wd_minmean_irg2[7]], 
                         "ws9":[ws_minmean_irg2[8],ws_log_minmean_irg2[8],wd_minmean_irg2[8]],
                         "ws10":[ws_minmean_irg2[9],ws_log_minmean_irg2[9],wd_minmean_irg2[9]],
                         "wind_size":[0,0,0],
                         "overlap":[0,0,0],
                         "search_area_size":[0,0,0],
                         "rmse":[0,0,0]}) 

  
    
    
def processInput(method, res_df, minmean_res_df, pos_vec_ret = (90,120),size_aroundIRG=3):
   
#for method in methods_lst:
    #method= "rmse"
    print(method)
    
    for j in range(0,len(ws_lst)):#
        print(j)
        window_size=ws_lst[j]
        overlap = ol_lst[j]
        search_area_size = sa_lst[j]
        

        #outpath = datapath+"tiv/method_"+method+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+"/"
    
        #if not os.path.exists(outpath):
            #os.makedirs(outpath)

        
        if method == "greyscale" or method =="rmse" or method =="ssim":
            try:
                
                for i in range(139,len(pertub)-6,6):
                    #print(i)
                    frame_a = pertub[i]
                    frame_b = pertub[i+6]
                
                
                    u1, v1= window_correlation_tiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                                           search_area_size_x=search_area_size, search_area_size_y=0, corr_method=method)
                    
                    
                    x1, y1 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
                    
                   
                    get_field_shape(image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
                    #frame_a.shape[0]/get_field_shape(image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )[0]
                    #frame_a.shape[1]/get_field_shape(image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )[1]
                    
                    #v1 = np.flipud(v1)
                    #u1 = np.flipud(u1)
                    #x1[0][9]
                    #frame_a.shape[0]-y1[9][0]
                    u1=np.reshape(np.flip(u1,0),(1,u1.shape[0],u1.shape[1]))
                    v1= np.reshape(np.flip(v1,0),(1,v1.shape[0],v1.shape[1]))
           
                    if i == 139:
                        uas =  copy.copy(u1)
                        vas = copy.copy(v1)
                       
                    else:
                        uas = np.append(uas,u1,0)
                        vas = np.append(vas,v1,0)
    
                    #
                    #streamplot(u1,v1*-1,X= x1, Y=y1, topo=frame_b,  vmin = -2, vmax = 2 )
                    #plt.show()
                    
                    
            except:
                
                print(method)
                print(j)
                print("<- j failed")
                pass
            try:
               
                
                
                vas_mean = np.nanmean(vas, axis=0)
                uas_mean = np.nanmean(uas, axis=0)
                uas[np.isnan(uas)] = 0
                vas[np.isnan(vas)] = 0
                
                #np.where(np.logical_and(, wd0<15))
                
                
                closest_pos_x = np.argmin(np.abs(x1[0]-pos_vec_ret[0]))
                closest_pos_y = np.argmin(np.abs(np.flip(y1[:,1])-pos_vec_ret[1]))
                
                
                
                #v_closest_pos = v[closest_pos]
                ws_mean = calcwindspeed(vas_mean, uas_mean)
                wd_mean = calcwinddirection(vas_mean,uas_mean)
                
                
                uas_cut = uas[:,closest_pos_y-size_aroundIRG:closest_pos_y+size_aroundIRG,closest_pos_x-size_aroundIRG:closest_pos_x+size_aroundIRG]
                vas_cut = vas[:,closest_pos_y-size_aroundIRG:closest_pos_y+size_aroundIRG,closest_pos_x-size_aroundIRG:closest_pos_x+size_aroundIRG]
                
                wd_cut_mean = wd_mean[closest_pos_y-size_aroundIRG:closest_pos_y+size_aroundIRG,closest_pos_x-size_aroundIRG:closest_pos_x+size_aroundIRG]
                ws_cut_mean = ws_mean[closest_pos_y-size_aroundIRG:closest_pos_y+size_aroundIRG,closest_pos_x-size_aroundIRG:closest_pos_x+size_aroundIRG]
                
    
                wd0 = wd_cut_mean - wd_mean_irg2
                wd0[wd0<-30] = 0
                wd0[wd0>30] = 0
               
                
                
                
                if np.mean(wd0) != 0:
                    print("yes")
                    # calc min mean of the single pixels and create rmse
                    not_0 = np.where(wd0 != 0)
                    
                    for i in range(0,len(not_0[0])):
                        act_uas = uas_cut[:,not_0[0][i],not_0[1][i]]
                        act_vas = vas_cut[:,not_0[0][i],not_0[1][i]]
                        act_wd_mean = wd_cut_mean[not_0[0][i],not_0[1][i]]
                        act_ws_mean = ws_cut_mean[not_0[0][i],not_0[1][i]]
                        act_uas_minmean = np.array(minmean(act_uas))
                        act_vas_minmean = np.array(minmean(act_vas))
                        
                        
                        
                                          
                        act_min_mean_wd = calcwinddirection(act_vas_minmean, act_uas_minmean)
                        act_min_mean_ws = calcwindspeed(act_uas_minmean*0.195, act_vas_minmean*0.195)
                        
                        act_rmse = rmse(act_min_mean_wd,wd_minmean_irg2)
                        
                        act_minmean_df = pd.DataFrame({"method":["method", "ws", "wd"], 
                             "ws1":[method,act_min_mean_ws[0],act_min_mean_wd[0]],
                             "ws2":[method,act_min_mean_ws[1],act_min_mean_wd[1]],
                             "ws3":[method,act_min_mean_ws[2],act_min_mean_wd[2]],
                             "ws4":[method,act_min_mean_ws[3],act_min_mean_wd[3]],
                             "ws5":[method,act_min_mean_ws[4],act_min_mean_wd[4]],
                             "ws6":[method,act_min_mean_ws[5],act_min_mean_wd[5]],
                             "ws7":[method,act_min_mean_ws[6],act_min_mean_wd[6]],           
                             "ws8":[method,act_min_mean_ws[7],act_min_mean_wd[7]], 
                             "ws9":[method,act_min_mean_ws[8],act_min_mean_wd[8]],
                             "ws10":[method,act_min_mean_ws[9],act_min_mean_wd[9]],
                             "wind_size":[window_size,window_size,window_size],
                             "overlap":[overlap,overlap,overlap],
                             "search_area_size":[search_area_size,search_area_size,search_area_size],
                             "rmse":[act_rmse,act_rmse,act_rmse]}) 
                        
                        
                        act_res_df = pd.DataFrame({"method":[method], 
                                                         "ws":[window_size],
                                                         "ol":[overlap],
                                                         "sa":[search_area_size],
                                                         "pos_y":[closest_pos_y-size_aroundIRG+not_0[0][i]],
                                                         "pos_x":[closest_pos_x-size_aroundIRG+not_0[1][i]],
                                                         "rmse":[act_rmse],
                                                         "mean_wd": [act_wd_mean],
                                                         "mean_ws": [act_ws_mean*0.195],
                                                         "diff_direc": [act_wd_mean-wd_mean_irg2],
                                                         "diff_ws": [(act_ws_mean*0.195)-ws_log_mean_irg2]})
                        
                        minmean_res_df = minmean_res_df.append(act_minmean_df, ignore_index = True)
                        res_df = res_df.append(act_res_df, ignore_index = True) 
                        
            except:
                print(method)
                print(j)
                print("<- j failed")
                pass         
                
                
            
            

            
                
        else:
            try:
                
                #testa = np.random.rand(pertub[i].shape[0],pertub[i].shape[1])
                #testb = np.random.rand(pertub[i].shape[0],pertub[i].shape[1])
                #testa[:,:] = 3
                #testb[:,:] = 3
                #testa[115:140,115:140] = 50
                #testb[140:165,140:165] = 50
                
                
                
                for i in range(139,len(pertub)-6,6):
                    #print(i)
                    frame_a = pertub[i]
                    frame_b = pertub[i+6]
                    
                    u, v = cross_correlation_aiv(frame_a, frame_b, window_size=window_size, overlap=overlap,
                                              dt=1, search_area_size=search_area_size, nfftx=None, 
                                              nffty=None,width=2, corr_method='fft', subpixel_method='gaussian')
                    
                    
                    x, y = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
                    
                    
                   
                    v = v*-1
                    ## vectorplot
                    #u = np.flipud(u)
                    #v = np.flipud(v)
                
                 
                    #plt.imshow(y)
                    #u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
                
                    
                    #plt.figure()
                    #plt.imshow(frame_a, vmin = -5, vmax=5)
                    #plt.quiver(x,y,u,v)
                    #plt.savefig(outpath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    #plt.close()
                    u=np.reshape(u,(1,u.shape[0],u.shape[1]))
                    v= np.reshape(v,(1,v.shape[0],v.shape[1]))
           
                    if i == 139:
                        uas =  copy.copy(u)
                        vas = copy.copy(v)
                       
                    else:
                        uas = np.append(uas,u,0)
                        vas = np.append(vas,v,0)
     
                    
                    
            except:
                print("CC")
                print(j)
                print("<- j failed")
                pass
            
            
            
            try:
                
                
                
                
                vas_mean = np.nanmean(vas, axis=0)
                uas_mean = np.nanmean(uas, axis=0)
                uas[np.isnan(uas)] = 0
                vas[np.isnan(vas)] = 0
                
                closest_pos_x = np.argmin(np.abs(x[0]-pos_vec_ret[0]))
                closest_pos_y = np.argmin(np.abs(np.flip(y[:,1])-pos_vec_ret[1]))
                
                
                
                #v_closest_pos = v[closest_pos]
                ws_mean = calcwindspeed(vas_mean,uas_mean)
                wd_mean = calcwinddirection(vas_mean,uas_mean)
                
                
                uas_cut = uas[:,closest_pos_y-size_aroundIRG:closest_pos_y+size_aroundIRG,closest_pos_x-size_aroundIRG:closest_pos_x+size_aroundIRG]
                vas_cut = vas[:,closest_pos_y-size_aroundIRG:closest_pos_y+size_aroundIRG,closest_pos_x-size_aroundIRG:closest_pos_x+size_aroundIRG]
                
                wd_cut_mean = wd_mean[closest_pos_y-size_aroundIRG:closest_pos_y+size_aroundIRG,closest_pos_x-size_aroundIRG:closest_pos_x+size_aroundIRG]
                ws_cut_mean = ws_mean[closest_pos_y-size_aroundIRG:closest_pos_y+size_aroundIRG,closest_pos_x-size_aroundIRG:closest_pos_x+size_aroundIRG]
                
    
                wd0 = wd_cut_mean - wd_mean_irg2
                wd0[wd0<-30] = 0
                wd0[wd0>30] = 0
               
                
                
                
                if np.mean(wd0) != 0:
                    print("yes")
                    # calc min mean of the single pixels and create rmse
                    
                    not_0 = np.where(wd0 != 0)
                    
                    for i in range(0,len(not_0[0])):
                        
                        act_uas = uas_cut[:,not_0[0][i],not_0[1][i]]
                        act_vas = vas_cut[:,not_0[0][i],not_0[1][i]]
                        
                        act_wd_mean = wd_cut_mean[not_0[0][i],not_0[1][i]]
                        act_ws_mean = ws_cut_mean[not_0[0][i],not_0[1][i]]
                        act_uas_minmean = np.array(minmean(act_uas))
                        act_vas_minmean = np.array(minmean(act_vas))
                        
                        
                        
                                          
                        act_min_mean_wd = calcwinddirection(act_vas_minmean,act_uas_minmean)
                        act_min_mean_ws = calcwindspeed(act_uas_minmean*0.195, act_vas_minmean*0.195)
                        
                        act_rmse = rmse(act_min_mean_wd,wd_minmean_irg2)
                        
                        act_minmean_df = pd.DataFrame({"method":["method","ws", "wd"], 
                             "ws1":[method,act_min_mean_ws[0],act_min_mean_wd[0]],
                             "ws2":[method,act_min_mean_ws[1],act_min_mean_wd[1]],
                             "ws3":[method,act_min_mean_ws[2],act_min_mean_wd[2]],
                             "ws4":[method,act_min_mean_ws[3],act_min_mean_wd[3]],
                             "ws5":[method,act_min_mean_ws[4],act_min_mean_wd[4]],
                             "ws6":[method,act_min_mean_ws[5],act_min_mean_wd[5]],
                             "ws7":[method,act_min_mean_ws[6],act_min_mean_wd[6]],           
                             "ws8":[method,act_min_mean_ws[7],act_min_mean_wd[7]], 
                             "ws9":[method,act_min_mean_ws[8],act_min_mean_wd[8]],
                             "ws10":[method,act_min_mean_ws[9],act_min_mean_wd[9]],
                             "wind_size":[window_size,window_size,window_size],
                             "overlap":[overlap,overlap,overlap],
                             "search_area_size":[search_area_size,search_area_size,search_area_size],
                             "rmse":[act_rmse,act_rmse,act_rmse]})  
                        
                        
                        act_res_df = pd.DataFrame({"method":[method], 
                                                         "ws":[window_size],
                                                         "ol":[overlap],
                                                         "sa":[search_area_size],
                                                         "pos_y":[closest_pos_y-size_aroundIRG+not_0[0][i]],
                                                         "pos_x":[closest_pos_x-size_aroundIRG+not_0[1][i]],
                                                         "rmse":[act_rmse],
                                                         "mean_wd": [act_wd_mean],
                                                         "mean_ws": [act_ws_mean*0.195],
                                                         "diff_direc": [act_wd_mean-wd_mean_irg2],
                                                         "diff_ws": [(act_ws_mean*0.195)-ws_log_mean_irg2]})
                        
                        minmean_res_df = minmean_res_df.append(act_minmean_df, ignore_index = True)
                        res_df = res_df.append(act_res_df, ignore_index = True) 
            except:
                print("CC")
                print(j)
                print("<- j failed")
                pass
                    
                
        
    return([res_df,minmean_res_df])



num_cores = 8

lst_dfs = Parallel(n_jobs=num_cores)(delayed(processInput)(method, res_df, minmean_res_df) for method in methods_lst )



for i in range(0, len(lst_dfs)):
    for j in range(0,len(lst_dfs[i])):
        lst_dfs[i][j].to_csv("/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/tiv/performance/"+str(i)+"_"+str(j)+".csv")
        

act_res_df.to_csv
rmse_df = lst_dfs[0]

minmean_df = lst_dfs[1]

lst_dfs[1][1]


plt.imshow()














import numpy as np

def rmse(m, o):
    m = np.array(m)
    o = np.array(o)
    return np.sqrt(np.mean((m-o)**2))
    





irg_ws = [0.25,	0.22,	0.2,	0.14,	0.27,	0.26,	0.25,	0.22,	0.11,	0.22]
    

CC_ws = [0.14,	0.05,	0.07,	0.12,	0.09,	0.09,	0.02,	0.01,	0.03,	0.07]    

SQM_ws = [0.55,	0.46,	0.39,	0.32,	0.26,	0.24,	0.46,	0.27,	0.22,	0.18]

gs_ws = [0.2,	0.28,	0.21,	0.49,	0.18,	0.16,	0.33,	0.35,	0.33,	0.32]

SSIM_ws = [0.09,	0.04,	0.11,	0.08,	0.33,	0.06,	0.04,	0.04,	0.08,	0.04]        
        
        
        
rmse(irg_ws,SQM_ws)        
      
rmse(irg_ws,CC_ws)        
 
rmse(irg_ws,SSIM_ws)        
                
     
rmse(irg_ws,gs_ws)        
 





















for m in range(0,len(ws_lst)):
    print("u mean:")
    print(np.mean(methods_u_pos_lst[3][m]))
    
    print("v mean:")
    print(np.mean(methods_v_pos_lst[3][m]))




# cc dataframe build
 
cc_IV_array = np.array([ws_lst,ol_lst,sa_lst])     
cc_IV_array = np.flipud(np.rot90(cc_IV_array))    
    
u_minmean = np.zeros((17,10))
v_minmean = np.zeros((17,10))
    


def calc_mean(lst):
    retlst = []
    for i in range(0,len(lst)):
        retlst.append(np.nanmean(lst[i]))
    return retlst
    

for m in range(0,len(ws_lst)):
    u_minmean[m,] = minmean(methods_u_pos_lst[0][m])
    v_minmean[m,] = minmean(methods_v_pos_lst[0][m])

u_minmean[:,0]
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


u_mean_irg1 = np.reshape(np.repeat(np.mean(irg1_subset.iloc[:,1]),17),(17,1))
v_mean_irg1 = np.reshape(np.repeat(np.mean(irg1_subset.iloc[:,2]),17),(17,1))

u_minmean_irg1 = np.zeros((17,10))

for i in range(0,17):
    u_minmean_irg1[i,:] = minmean(irg1_subset.iloc[:,1], 1200) 
v_minmean_irg1 = np.zeros((17,10))
for i in range(0,17):
    v_minmean_irg1[i,:] = minmean(irg1_subset.iloc[:,2], 1200) 





u_mean_irg2 = np.reshape(np.repeat(np.mean(irg2_subset.iloc[:,1]),17),(17,1))
v_mean_irg2 = np.reshape(np.repeat(np.mean(irg2_subset.iloc[:,2]),17),(17,1))

u_minmean_irg2 = np.zeros((17,10))
for i in range(0,17):
    u_minmean_irg2[i,:] = minmean(irg2_subset.iloc[:,1], 1200) 
v_minmean_irg2 = np.zeros((17,10))
for i in range(0,17):
    v_minmean_irg2[i,:] = minmean(irg2_subset.iloc[:,2], 1200) 

cc_IV_array= np.concatenate((cc_IV_array, u_minmean*0.195, v_minmean*0.195, u_mean*0.195, v_mean*0.195, 
                             u_3x3_minmean*0.195, v_3x3_minmean*0.195, u_3x3_mean*0.195, v_3x3_mean*0.195,
                             u_5x5_minmean*0.195, v_5x5_minmean*0.195, u_5x5_mean*0.195, v_5x5_mean*0.195, u_minmean_irg1, v_minmean_irg1, u_mean_irg1, v_mean_irg1, u_minmean_irg2, v_minmean_irg2, u_mean_irg2, v_mean_irg2), axis=1)


cc_IV_df = pd.DataFrame(cc_IV_array, columns=["window_size", "overlap", "search_area_size","u_minmean1", "u_minmean2", "u_minmean3", "u_minmean4", "u_minmean5", "u_minmean6", "u_minmean7", "u_minmean8", "u_minmean9", "u_minmean10",
                                    "v_minmean1", "v_minmean2", "v_minmean3", "v_minmean4", "v_minmean5", "v_minmean6", "v_minmean7", "v_minmean8", "v_minmean9", "v_minmean10", "u_mean", "v_mean", 
                            "u_3x3_minmean1", "u_3x3_minmean2", "u_3x3_minmean3", "u_3x3_minmean4", "u_3x3_minmean5", "u_3x3_minmean6", "u_3x3_minmean7", "u_3x3_minmean8", "u_3x3_minmean9", "u_3x3_minmean10",
                            "v_3x3_minmean1", "v_3x3_minmean2", "v_3x3_minmean3", "v_3x3_minmean4", "v_3x3_minmean5", "v_3x3_minmean6", "v_3x3_minmean7", "v_3x3_minmean8", "v_3x3_minmean9", "v_3x3_minmean10", "u_3x3_mean", "v_3x3_mean",
                            "u_5x5_minmean1", "u_5x5_minmean2", "u_5x5_minmean3", "u_5x5_minmean4", "u_5x5_minmean5", "u_5x5_minmean6", "u_5x5_minmean7", "u_5x5_minmean8", "u_5x5_minmean9", "u_5x5_minmean10",
                            "v_5x5_minmean1", "v_5x5_minmean2", "v_5x5_minmean3", "v_5x5_minmean4", "v_5x5_minmean5", "v_5x5_minmean6", "v_5x5_minmean7", "v_5x5_minmean8", "v_5x5_minmean9", "v_5x5_minmean10", "u_5x5_mean", "v_5x5_mean",
                            "u_minmean_irg1_1", "u_minmean_irg1_2", "u_minmean_irg1_3", "u_minmean_irg1_4", "u_minmean_irg1_5", "u_minmean_irg1_6", "u_minmean_irg1_7", "u_minmean_irg1_8", "u_minmean_irg1_9", "u_minmean_irg1_10",
                            "v_minmean_irg1_1", "v_minmean_irg1_2", "v_minmean_irg1_3", "v_minmean_irg1_4", "v_minmean_irg1_5", "v_minmean_irg1_6", "v_minmean_irg1_7", "v_minmean_irg1_9", "v_minmean_irg1_9", "v_minmean_irg1_10", "u_mean_irg1", "v_mean_irg1",
                            "u_minmean_irg2_1", "u_minmean_irg2_2", "u_minmean_irg2_3", "u_minmean_irg2_4", "u_minmean_irg2_5", "u_minmean_irg2_6", "u_minmean_irg2_7", "u_minmean_irg2_8", "u_minmean_irg2_9", "u_minmean_irg2_10",
                            "v_minmean_irg2_1", "v_minmean_irg2_2", "v_minmean_irg2_3", "v_minmean_irg2_4", "v_minmean_irg2_5", "v_minmean_irg2_6", "v_minmean_irg2_7", "v_minmean_irg2_9", "v_minmean_irg2_9", "v_minmean_irg2_10", "u_mean_irg2", "v_mean_irg2"])


cc_IV_df.to_csv("/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/cc_IV_df.csv")

 
# gs dataframe build


for pos_gs in range(1,4):
    print(methods_lst[pos_gs])
    gs_IV_array = np.array([ws_lst,ol_lst,sa_lst])     
    gs_IV_array = np.flipud(np.rot90(gs_IV_array))    
        
    u_minmean = np.zeros((17,10))
    v_minmean = np.zeros((17,10))
     
    
    for m in range(0,len(ws_lst)):
        try:
            u_minmean[m,] = minmean(methods_u_pos_lst[pos_gs][m])
            v_minmean[m,] = minmean(methods_v_pos_lst[pos_gs][m])
        except:
            u_minmean[m,] = np.repeat(0,10).tolist()
            v_minmean[m,] = np.repeat(0,10).tolist()
    
    
    u_mean = np.reshape(np.array(calc_mean(methods_u_pos_lst[pos_gs])),(17,1))
    v_mean = np.reshape(np.array(calc_mean(methods_v_pos_lst[pos_gs])),(17,1))
    
    
       
    u_3x3_minmean = np.zeros((17,10))
    v_3x3_minmean = np.zeros((17,10))
        
       
    
    for m in range(0,len(ws_lst)):
        try:
            u_3x3_minmean[m,] = minmean(methods_3x3_u_lst[pos_gs][m])
            v_3x3_minmean[m,] = minmean(methods_3x3_v_lst[pos_gs][m])
        except:
            u_3x3_minmean[m,] = np.repeat(0,10).tolist()
            v_3x3_minmean[m,] = np.repeat(0,10).tolist()
    
    
    u_3x3_mean = np.reshape(np.array(calc_mean(methods_3x3_u_lst[pos_gs])),(17,1))
    v_3x3_mean = np.reshape(np.array(calc_mean(methods_3x3_v_lst[pos_gs])),(17,1))
    
       
    u_5x5_minmean = np.zeros((17,10))
    v_5x5_minmean = np.zeros((17,10))
        
       
    
    for m in range(0,len(ws_lst)):
        try:
            u_5x5_minmean[m,] = minmean(methods_5x5_u_lst[pos_gs][m])
            v_5x5_minmean[m,] = minmean(methods_5x5_v_lst[pos_gs][m])
        except:
            u_5x5_minmean[m,] = np.repeat(0,10).tolist()
            v_5x5_minmean[m,] = np.repeat(0,10).tolist()
    
    
    u_5x5_mean = np.reshape(np.array(calc_mean(methods_5x5_u_lst[pos_gs])),(17,1))
    v_5x5_mean = np.reshape(np.array(calc_mean(methods_5x5_v_lst[pos_gs])),(17,1))
    
    
    
    u_mean_irg1 = np.reshape(np.repeat(np.mean(irg1_subset.iloc[:,1]),17),(17,1))
    v_mean_irg1 = np.reshape(np.repeat(np.mean(irg1_subset.iloc[:,2]),17),(17,1))
    
    
    u_mean_irg2 = np.reshape(np.repeat(np.mean(irg2_subset.iloc[:,1]),17),(17,1))
    v_mean_irg2 = np.reshape(np.repeat(np.mean(irg2_subset.iloc[:,2]),17),(17,1))
    
    
    
    gs_IV_array= np.concatenate((gs_IV_array, u_minmean*0.195, v_minmean*0.195, u_mean*0.195, v_mean*0.195, 
                                 u_3x3_minmean*0.195, v_3x3_minmean*0.195, u_3x3_mean*0.195, v_3x3_mean*0.195,
                                 u_5x5_minmean*0.195, v_5x5_minmean*0.195, u_5x5_mean*0.195, v_5x5_mean*0.195, u_minmean_irg1, v_minmean_irg1, u_mean_irg1, v_mean_irg1, u_minmean_irg2, v_minmean_irg2, u_mean_irg2, v_mean_irg2), axis=1)
    
    
    gs_IV_df = pd.DataFrame(gs_IV_array, columns=["window_size", "overlap", "search_area_size","u_minmean1", "u_minmean2", "u_minmean3", "u_minmean4", "u_minmean5", "u_minmean6", "u_minmean7", "u_minmean8", "u_minmean9", "u_minmean10",
                                        "v_minmean1", "v_minmean2", "v_minmean3", "v_minmean4", "v_minmean5", "v_minmean6", "v_minmean7", "v_minmean8", "v_minmean9", "v_minmean10", "u_mean", "v_mean", 
                                "u_3x3_minmean1", "u_3x3_minmean2", "u_3x3_minmean3", "u_3x3_minmean4", "u_3x3_minmean5", "u_3x3_minmean6", "u_3x3_minmean7", "u_3x3_minmean8", "u_3x3_minmean9", "u_3x3_minmean10",
                                "v_3x3_minmean1", "v_3x3_minmean2", "v_3x3_minmean3", "v_3x3_minmean4", "v_3x3_minmean5", "v_3x3_minmean6", "v_3x3_minmean7", "v_3x3_minmean8", "v_3x3_minmean9", "v_3x3_minmean10", "u_3x3_mean", "v_3x3_mean",
                                "u_5x5_minmean1", "u_5x5_minmean2", "u_5x5_minmean3", "u_5x5_minmean4", "u_5x5_minmean5", "u_5x5_minmean6", "u_5x5_minmean7", "u_5x5_minmean8", "u_5x5_minmean9", "u_5x5_minmean10",
                                "v_5x5_minmean1", "v_5x5_minmean2", "v_5x5_minmean3", "v_5x5_minmean4", "v_5x5_minmean5", "v_5x5_minmean6", "v_5x5_minmean7", "v_5x5_minmean8", "v_5x5_minmean9", "v_5x5_minmean10", "u_5x5_mean", "v_5x5_mean",
                                "u_minmean_irg1_1", "u_minmean_irg1_2", "u_minmean_irg1_3", "u_minmean_irg1_4", "u_minmean_irg1_5", "u_minmean_irg1_6", "u_minmean_irg1_7", "u_minmean_irg1_8", "u_minmean_irg1_9", "u_minmean_irg1_10",
                                "v_minmean_irg1_1", "v_minmean_irg1_2", "v_minmean_irg1_3", "v_minmean_irg1_4", "v_minmean_irg1_5", "v_minmean_irg1_6", "v_minmean_irg1_7", "v_minmean_irg1_9", "v_minmean_irg1_9", "v_minmean_irg1_10", "u_mean_irg1", "v_mean_irg1",
                                "u_minmean_irg2_1", "u_minmean_irg2_2", "u_minmean_irg2_3", "u_minmean_irg2_4", "u_minmean_irg2_5", "u_minmean_irg2_6", "u_minmean_irg2_7", "u_minmean_irg2_8", "u_minmean_irg2_9", "u_minmean_irg2_10",
                                "v_minmean_irg2_1", "v_minmean_irg2_2", "v_minmean_irg2_3", "v_minmean_irg2_4", "v_minmean_irg2_5", "v_minmean_irg2_6", "v_minmean_irg2_7", "v_minmean_irg2_9", "v_minmean_irg2_9", "v_minmean_irg2_10", "u_mean_irg2", "v_mean_irg2"])
    
    
    gs_IV_df.to_csv("/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/"+methods_lst[pos_gs]+"_IV_df.csv")

    

