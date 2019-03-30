#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:51:30 2019

@author: benjamin
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import colors, cm
from joblib import Parallel, delayed
import multiprocessing
from TST_fun import *
import datetime
from math import sqrt
import copy
import scipy as sio



datapath = "/home/benjamin/Met_ParametersTST/PALMS/data/"
outpath = datapath+"tiv/"

file = h5py.File(datapath+'cbl_surf.nc','r')
Tb = file.get("tsurf*_xy")
Tb = np.array(Tb)
Tb = np.reshape(Tb,(3600,256,512))


u_model = file.get("u_xy")
v_model = file.get("v_xy")
u_model = np.reshape(u_model,(3600,256,512))
v_model = np.reshape(v_model,(3600,256,512))



# calculate pertubations

#Tb_pertub = create_tst_pertubations_mm(Tb, moving_mean_size = 60)
#writeNetCDF(outpath, "cbl_surf_pertub.nc", "Tb_pertub", Tb_pertub)





time_a = datetime.datetime.now()



interwindowsize_x_lst = [5, 7, 9, 13, 17]
interwindowsize_y_lst = [5, 7, 9, 13, 17]

max_displacement_lst = [5,7,13]
interval_lst = [1,2,3]
counter = 0

mean_error_tiv_lst = []
mean_error_tiv_rmse_lst= []
mean_error_tiv_ssim_lst= []
mean_error_tiv_greyscalediag_lst= []

rmse_error_tiv_lst= []
rmse_error_tiv_rmse_lst= []
rmse_error_tiv_ssim_lst= []
rmse_error_tiv_greyscalediag_lst= []

settings_lst = []

my_dpi = 100

#

#for a in range(0,len(interwindowsize_x_lst)):
#               
#    print(counter)
#   
#    inter_window_size_x = interwindowsize_x_lst[a]
#    for b in range(0,len(interwindowsize_y_lst)):
#        
#        inter_window_size_y = interwindowsize_y_lst[b]
#        for c in range(0,len(max_displacement_lst)):
#            
#            max_displacement = max_displacement_lst[c]
#            for d in range(0,len(interval_lst)):
#                
#                
                #interval = interval_lst[d]
                
                
                #counter +=1

                inter_window_size_x = 11
                inter_window_size_y = 11
                max_displacement = 9
                inter_window_size_dif_x = int((inter_window_size_x-1)/2) 
           
                inter_window_size_dif_y = int((inter_window_size_y-1)/2) 
                
                #rand_window = np.random.rand(inter_window_size_y+1, inter_window_size_x+1)
                #plt.imshow(rand_window)
                
                interval = 6
                num_cores = multiprocessing.cpu_count()-4
                begin_value_images = 3540
                
                end_of_u = 512
                end_of_v = 256
                begin_of_v = 0
                begin_of_u = 0
                begin_value_images = 3540
                end_value_images = begin_value_images+interval+1
                
                
                iter_lst = []
                for i in range(-max_displacement,max_displacement):
                    # displace in x direction
                    for k in range(-max_displacement,max_displacement):
                        #displace in y direction
                        iter_lst.append((i,k))
                
                
                
                
                
                #Tb_pertub= Tb_pertub[begin_value_images:end_value_images]
                #Tb_pertub = Tb_pertub[:,begin_of_v:end_of_v,begin_of_u:end_of_u]
                Tb_subset = Tb[begin_value_images:end_value_images]
                u_subset = u_model[begin_value_images:end_value_images]
                v_subset = v_model[begin_value_images:end_value_images]
                
                windspeed = copy.copy(u_subset)
                windspeed[0] = calcwindspeed(v_subset[0],u_subset[0])
                windspeed[1] = calcwindspeed(v_subset[1],u_subset[1])
                
                #try:
                    out_u_v = Parallel(n_jobs=num_cores)(delayed(calc_TIV)(n, u_subset, iterx = max_displacement+inter_window_size_dif_x -1,itery = max_displacement+inter_window_size_dif_y -1, 
                                       interval = interval, max_displacement=max_displacement, inter_window_size_dif_x = inter_window_size_dif_x, 
                                       inter_window_size_dif_y = inter_window_size_dif_y, iter_lst = iter_lst,
                                       inter_window_size_x = inter_window_size_x, inter_window_size_y = inter_window_size_y, method = "greyscale") for n in range(begin_value_images-begin_value_images,end_value_images-begin_value_images-interval))
                    
                    
                    time_b = datetime.datetime.now()
                    print(time_b-time_a)
                    
                    out_u_v_rmse = Parallel(n_jobs=num_cores)(delayed(calc_TIV)(n, u_subset, iterx = max_displacement+inter_window_size_dif_x -1,itery = max_displacement+inter_window_size_dif_y -1, 
                                       interval = interval, max_displacement=max_displacement, inter_window_size_dif_x = inter_window_size_dif_x, 
                                       inter_window_size_dif_y = inter_window_size_dif_y, iter_lst = iter_lst,
                                       inter_window_size_x = inter_window_size_x, inter_window_size_y = inter_window_size_y, method = "rmse") for n in range(begin_value_images-begin_value_images,end_value_images-begin_value_images-interval))
                    
                    time_c = datetime.datetime.now()
                    print(time_c-time_b)
                    
                    #out_u_v_ssim = Parallel(n_jobs=num_cores)(delayed(calc_TIV_sqdif)(n, Tb_subset, iterx = max_displacement+inter_window_size_dif -1,itery = max_displacement+inter_window_size_dif -1, interval = interval, max_displacement=max_displacement, inter_window_size = inter_window_size, inter_window_size_dif=inter_window_size_dif, iter_lst = iter_lst) for n in range(begin_value_images-begin_value_images,end_value_images-begin_value_images-interval))
                    
                    
                    out_u_v_ssim = Parallel(n_jobs=num_cores)(delayed(calc_TIV)(n, u_subset, iterx = max_displacement+inter_window_size_dif_x -1,itery = max_displacement+inter_window_size_dif_y -1, 
                                       interval = interval, max_displacement=max_displacement, inter_window_size_dif_x = inter_window_size_dif_x, 
                                       inter_window_size_dif_y = inter_window_size_dif_y, iter_lst = iter_lst,
                                       inter_window_size_x = inter_window_size_x, inter_window_size_y = inter_window_size_y, method = "ssim") for n in range(begin_value_images-begin_value_images,end_value_images-begin_value_images-interval))
                    
                    time_d = datetime.datetime.now()
                    print(time_d-time_c)
                    
                    
    
    
    
    
    #################################### Windspeed
    
    
                    plt.imshow(out_u_v[0][0], vmin=-7, vmax=7)
                    plt.colorbar()
                   
                    
                    
                    U = out_u_v[0][0]
                    V = out_u_v[0][1]
                    X = np.arange(0, U.shape[1], 1)
                    Y = np.arange(0, U.shape[0], 1)
                    stream_plot_rmse = streamplot(U, V, vmin=-4, vmax=4)
                    plt.quiver( X, Y, U, V )
                    
                    
                    
                    tiv_ws = calcwindspeed(out_u_v[0][1],out_u_v[0][0])
                    tiv_ws = tiv_ws[(max_displacement+inter_window_size_dif_y):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif_y),(max_displacement+inter_window_size_dif_x):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif_x)]
                    tiv_ws = tiv_ws/interval
                    plt.imshow(tiv_ws, vmin=0, vmax=7)
                    plt.colorbar()
                    title_name = "TIV windspeed X:"+str(inter_window_size_x)+" Y:"+str(inter_window_size_y)+" MaxDisp:"+str(max_displacement)+" Interv:"+str(interval)
                    
                    plt.title(title_name)
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/greyscale_tiv/"+title_name.replace(" ","_")+".png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
      
                    
                    tiv_ws_rmse = calcwindspeed(out_u_v_rmse[0][1],out_u_v_rmse[0][0])
                    tiv_ws_rmse = tiv_ws_rmse[(max_displacement+inter_window_size_dif_y):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif_y),(max_displacement+inter_window_size_dif_x):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif_x)]
                    tiv_ws_rmse = tiv_ws_rmse/interval
                    plt.imshow(tiv_ws_rmse, vmin=0, vmax=7)
                    plt.colorbar()
                    title_name = "RMSE TIV windspeed X:"+str(inter_window_size_x)+" Y:"+str(inter_window_size_y)+" MaxDisp:"+str(max_displacement)+" Interv:"+str(interval)
                    
                    plt.title(title_name)
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/rmse_tiv/"+title_name.replace(" ","_")+".png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    
                    
                    tiv_ws_ssim = calcwindspeed(out_u_v_ssim[0][1],out_u_v_ssim[0][0])
                    tiv_ws_ssim = tiv_ws_ssim[(max_displacement+inter_window_size_dif_y):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif_y),(max_displacement+inter_window_size_dif_x):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif_x)]
                    tiv_ws_ssim = tiv_ws_ssim/interval
                    plt.imshow(tiv_ws_ssim, vmin=0, vmax=7)
                    plt.colorbar()
                    title_name = "SSIM TIV windspeed X:"+str(inter_window_size_x)+" Y:"+str(inter_window_size_y)+" MaxDisp:"+str(max_displacement)+" Interv:"+str(interval)
                    plt.title(title_name)
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/ssim_tiv/"+title_name.replace(" ","_")+".png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                   
                    
                    model_ws1 = calcwindspeed(v_subset[0],u_subset[0])
                    model_ws2 = calcwindspeed(v_subset[1],u_subset[1])
                    model_ws = (model_ws1+model_ws2)/2
                    model_ws = model_ws[(max_displacement+inter_window_size_dif_y):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif_y),(max_displacement+inter_window_size_dif_x):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif_x)]
                    plt.imshow(model_ws, vmin=0, vmax=7)
                    plt.colorbar()
                    plt.title("Mean-WindInterval1")
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/Mean-WindInterval1.png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                                        
                                        
                    
                    breaks=[-2, -1.8, -1.6, -1.4 ,-1.2, -1, -0.8 ,-0.6 ,-0.4, -0.2,0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4,4.6,4.8,5,5.2,5.4,5.6,5.8,6]
                    plt.hist(np.round((model_ws-tiv_ws).flatten(),1), bins=breaks, fc=(0, 0, 0.5, 0.5))
                    title_name = "Hist Dif Model - TIV Greyscale X:"+str(inter_window_size_x)+" Y:"+str(inter_window_size_y)+" MaxDisp:"+str(max_displacement)+" Interv:"+str(interval)
                    plt.title(title_name)
                    plt.text(1,2000, s= "Mean Dif: "+str(np.round(np.mean(model_ws-tiv_ws),2)) )
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/greyscale_tiv/"+title_name.replace(" ","_")+".png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    
                    breaks=[-2, -1.8, -1.6, -1.4 ,-1.2, -1, -0.8 ,-0.6 ,-0.4, -0.2,0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4,4.6,4.8,5,5.2,5.4,5.6,5.8,6]
                    plt.hist(np.round((model_ws-tiv_ws_rmse).flatten(),1), bins=breaks, fc=(0, 1, 0, 0.6))
                    title_name = "Hist Dif Model - TIV RMSE X:"+str(inter_window_size_x)+" Y:"+str(inter_window_size_y)+" MaxDisp:"+str(max_displacement)+" Interv:"+str(interval)
                    plt.title(title_name)
                    plt.text(1,2000, s= "Mean Dif: "+str(np.round(np.mean(model_ws-tiv_ws_rmse),2)) )
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/rmse_tiv/"+title_name.replace(" ","_")+".png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    breaks=[-2, -1.8, -1.6, -1.4 ,-1.2, -1, -0.8 ,-0.6 ,-0.4, -0.2,0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4,4.6,4.8,5,5.2,5.4,5.6,5.8,6]
                    plt.hist(np.round((model_ws-tiv_ws_ssim).flatten(),1), bins=breaks)
                    title_name = "Hist Dif Model - TIV SSIM X:"+str(inter_window_size_x)+" Y:"+str(inter_window_size_y)+" MaxDisp:"+str(max_displacement)+" Interv:"+str(interval)
                    plt.title(title_name)
                    plt.text(1,2000, s= "Mean Dif: "+str(np.round(np.mean(model_ws-tiv_ws_ssim),2)) )
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/ssim_tiv/"+title_name.replace(" ","_")+".png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    





                    plt.imshow(model_ws-tiv_ws)
                    plt.colorbar()
                    title_name = "Dif Model - TIV X:"+str(inter_window_size_x)+" Y:"+str(inter_window_size_y)+" MaxDisp:"+str(max_displacement)+" Interv:"+str(interval)
                    plt.title(title_name+"\nRMSE:"+str(round(np.sqrt(np.mean((model_ws-tiv_ws)**2)),3)))
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/greyscale_tiv/"+title_name.replace(" ","_")+".png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()


                    plt.imshow(model_ws-tiv_ws_rmse)
                    plt.colorbar()
                    title_name = "Dif Model - TIV RMSE X:"+str(inter_window_size_x)+" Y:"+str(inter_window_size_y)+" MaxDisp:"+str(max_displacement)+" Interv:"+str(interval)
                    plt.title(title_name+"\nRMSE:"+str(round(np.sqrt(np.mean((model_ws-tiv_ws_rmse)**2)),3)))
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/rmse_tiv/"+title_name.replace(" ","_")+".png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()

                    plt.imshow(model_ws-tiv_ws_ssim)
                    plt.colorbar()
                    title_name = "Dif Model - TIV SSIM X:"+str(inter_window_size_x)+" Y:"+str(inter_window_size_y)+" MaxDisp:"+str(max_displacement)+" Interv:"+str(interval)
                    plt.title(title_name+"\nRMSE:"+str(round(np.sqrt(np.mean((model_ws-tiv_ws_ssim)**2)),3)))
                    filename = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/ssim_tiv/"+title_name.replace(" ","_")+".png"
                    plt.savefig(filename,dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()

                    rmse_error_tiv_lst.append(round(np.sqrt(np.mean((model_ws-tiv_ws)**2)),3))  
                    rmse_error_tiv_rmse_lst.append(round(np.sqrt(np.mean((model_ws-tiv_ws_rmse)**2)),3))
                    rmse_error_tiv_ssim_lst.append(round(np.sqrt(np.mean((model_ws-tiv_ws_ssim)**2)),3))
                    
                    #rmse_error_tiv_greyscalediag_lst.append(round(np.sqrt(np.mean((model_ws-tiv_ws_greydiag)**2)),3))
                    
                    
                    
                    
                    
                    mean_error_tiv_lst.append(np.round(np.mean(model_ws-tiv_ws),2))
                    mean_error_tiv_rmse_lst.append(np.round(np.mean(model_ws-tiv_ws_rmse),2))
                    mean_error_tiv_ssim_lst.append(np.round(np.mean(model_ws-tiv_ws_ssim),2))
                    #mean_error_tiv_greyscalediag_lst.append(np.round(np.mean(model_ws-tiv_ws_greydiag),2))
                    settings_lst.append([inter_window_size_x,inter_window_size_y,max_displacement,interval])
                
                
                except:
                    pass




filename= "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test/"
np.savetxt("filename"+"RMSE_model_ws-tiv_ws.csv", rmse_error_tiv_lst, delimiter=",")
np.savetxt("filename"+"RMSE_model_ws-tiv_ws_rmse.csv", rmse_error_tiv_rmse_lst, delimiter=",")
np.savetxt("filename"+"RMSE_model_ws-tiv_ws_ssim.csv", rmse_error_tiv_ssim_lst, delimiter=",")

np.savetxt("filename"+"ME_model_ws-tiv_ws.csv", mean_error_tiv_lst, delimiter=",")
np.savetxt("filename"+"ME_model_ws-tiv_ws_rmse.csv", mean_error_tiv_rmse_lst, delimiter=",")
np.savetxt("filename"+"ME_model_ws-tiv_ws_ssim.csv", mean_error_tiv_ssim_lst, delimiter=",")

np.savetxt("filename"+"settings_lst.csv", settings_lst, delimiter=",")



plt.imshow(model_ws, vmin=0, vmax=7)
plt.colorbar()




streamplot(u[0], v[0])

plt.imshow(out_u_v[0][0])
plt.colorbar()

plt.imshow(out_u_v_rmse[0][0])
plt.colorbar()

plt.imshow(out_u_v_ssim[0][0])
plt.colorbar()

plt.imshow(out_u_v_greydiag[0][0])
plt.colorbar()

plt.imshow(Tb_subset[0]-Tb_subset[1])
plt.colorbar()






##################### Winddirection



tiv_wd = calcwinddirection(out_u_v[0][1],out_u_v[0][0])
tiv_wd = tiv_wd[(max_displacement+inter_window_size_dif_y):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif_y),(max_displacement+inter_window_size_dif_x):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif_x)]
plt.imshow(tiv_wd)
plt.colorbar()




model_wd = calcwinddirection(v_subset[0],u_subset[0])
model_wd = model_wd[(max_displacement+inter_window_size_dif_y):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif_y),(max_displacement+inter_window_size_dif_x):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif_x)]
plt.imshow(model_wd)
plt.colorbar()



plt.imshow(model_wd-tiv_wd)
plt.colorbar()































from scipy.ndimage import uniform_filter
plt.imshow(uniform_filter(Tb_subset[0], size=7))
plt.colorbar()



plt.imshow(u[3447])
plt.colorbar()


z = 0
x = 0


pertub_cut = Tb_subset[:,(max_displacement+inter_window_size_dif):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif),(max_displacement+inter_window_size_dif):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif)]

writeNetCDF(outpath,"Tb_pertub_cut_"+str(begin_value_images)+"-"+str(end_value_images)+".nc", "Tb_pertub",pertub_cut)

out_v_stack = np.zeros(pertub_cut.shape)
out_u_stack = np.zeros(pertub_cut.shape)
for o in range(0,len(out_u_v)):
    out_u_stack[o] = out_u_v[o][0,(max_displacement+inter_window_size_dif):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif),(max_displacement+inter_window_size_dif):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif)]
    out_v_stack[o] = out_u_v[o][1,(max_displacement+inter_window_size_dif):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif),(max_displacement+inter_window_size_dif):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif)]
    


out_v_stack_rmse = np.zeros(pertub_cut.shape)
out_u_stack_rmse = np.zeros(pertub_cut.shape)
for o in range(0,len(out_u_v)):
    out_u_stack_rmse[o] = out_u_v_rmse[o][0,(max_displacement+inter_window_size_dif):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif),(max_displacement+inter_window_size_dif):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif)]
    out_v_stack_rmse[o] = out_u_v_rmse[o][1,(max_displacement+inter_window_size_dif):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif),(max_displacement+inter_window_size_dif):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif)]
    



#output_file_u = out_u_v[z][x,(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]

#x = 2
#output_file_u_ttest = out_u_v[z][x,(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]


#x = 1
#output_file_v = out_u_v[z][x,(max_displacement+inter_window_size_dif):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif),(max_displacement+inter_window_size_dif):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif)]

#x = 3
#output_file_v_ttest = out_u_v[z][x,(max_displacement+inter_window_size):end_of_v-begin_of_v-(max_displacement+inter_window_size),(max_displacement+inter_window_size):end_of_u-begin_of_u-(max_displacement+inter_window_size)]

outname_u = "u_TIV_IW"+str(inter_window_size)+"_interval"+str(interval)+"_maxdisp"+str(max_displacement)+"_"+str(begin_value_images)+"-"+str(end_value_images)+".nc"
outname_v = "v_TIV_IW"+str(inter_window_size)+"_interval"+str(interval)+"_maxdisp"+str(max_displacement)+"_"+str(begin_value_images)+"-"+str(end_value_images)+".nc"


outname_u_rmse = "u_TIV_rmse_IW"+str(inter_window_size)+"_interval"+str(interval)+"_maxdisp"+str(max_displacement)+"_"+str(begin_value_images)+"-"+str(end_value_images)+".nc"
outname_v_rmse = "v_TIV_rmse_IW"+str(inter_window_size)+"_interval"+str(interval)+"_maxdisp"+str(max_displacement)+"_"+str(begin_value_images)+"-"+str(end_value_images)+".nc"


writeNetCDF(outpath, outname_u, "u",out_u_stack)

writeNetCDF(outpath, outname_v, "v",out_v_stack)


writeNetCDF(outpath, outname_u_rmse, "u",out_u_stack_rmse)

writeNetCDF(outpath, outname_v_rmse, "v",out_v_stack_rmse)

##########################################################################################################
# GREY SCALE

v_new_file = outpath+outname_v
u_new_file = outpath+outname_u

file_v = h5py.File(v_new_file,'r')
file_u = h5py.File(u_new_file,'r')
v = file_v.get("v")
v = np.array(v)
u = file_u.get("u")
u = np.array(u)





stream_plot_grey = streamplot(u[0], v[0])



u_model = file.get("u_xy")
v_model = file.get("v_xy")
u_model = np.reshape(u_model,(3600,256,512))
v_model = np.reshape(v_model,(3600,256,512))




plt.imshow(v[0])
plt.colorbar()


plt.imshow(calcwinddirection(V,U)) 
plt.colorbar()


tiv_ws = calcwindspeed(v[0]*2,u[0]*2)

plt.imshow(tiv_ws)
plt.colorbar()

plt.imshow(calcwindspeed(V*2,U*2)) 
plt.colorbar()



np.sqrt(np.mean((rolling_wind_arr_reshaped-rep_inter_wind1)**2,(1,2)))
###############################################################################################

# RMSE


v_new_file = outpath+outname_u_rmse
u_new_file = outpath+outname_v_rmse

file_v = h5py.File(v_new_file,'r')
file_u = h5py.File(u_new_file,'r')
v_rmse = file_v.get("v")
v_rmse = np.array(v)
u_rmse = file_u.get("u")
u_rmse = np.array(u)

u_rmse_smooth = sio.ndimage.filters.uniform_filter(u_rmse,size = 7,mode='constant')
u_rmse_gaus = sio.ndimage.filters.gaussian_filter(u_rmse,sigma=2, mode='wrap')
v_rmse_gaus = sio.ndimage.filters.gaussian_filter(v_rmse,sigma=2, mode='wrap')


stream_plot_rmse = streamplot(u_rmse_gaus[0]*2, v_rmse_gaus[0]*2, vmin=-4, vmax=4)

plt.imshow(u_rmse_gaus[0])
plt.colorbar()


plt.imshow(calcwinddirection(V,U)) 
plt.colorbar()


tiv_ws = calcwindspeed(v_rmse_gaus[0]*2, u_rmse_gaus[0]*2)

plt.imshow(tiv_ws)
plt.colorbar()

plt.imshow(calcwindspeed(V*2,U*2)) 
plt.colorbar()





###############################################################################################

# original

u_model_cut = u_model[3447,(max_displacement+inter_window_size_dif):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif),(max_displacement+inter_window_size_dif):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif)]
v_model_cut = v_model[3447,(max_displacement+inter_window_size_dif):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif),(max_displacement+inter_window_size_dif):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif)]


streamplot_model = streamplot(u_model_cut, v_model_cut, vmin=-4, vmax=4)
np.std(u_model_cut)
    
plt.imshow(u_model[3349])
plt.colorbar()

plt.imshow(calcwinddirection(v_model_cut,u_model_cut)) 
plt.colorbar()



plt.imshow(calcwindspeed(v_model_cut,u_model_cut)) 
plt.colorbar()
model_ws = calcwindspeed(v_model_cut,u_model_cut)

plt.imshow(u_model_cut) 
plt.colorbar()


plt.imshow(u_model[3348,(max_displacement+inter_window_size_dif):end_of_v-begin_of_v-(max_displacement+inter_window_size_dif),(max_displacement+inter_window_size_dif):end_of_u-begin_of_u-(max_displacement+inter_window_size_dif)]
)
plt.colorbar()

plt.imshow(tiv_ws-model_ws)
plt.colorbar()
plt.imshow(Tb[3347])
plt.colorbar()
plt.imshow((Tb[3348]-Tb[3347])*100)
plt.colorbar()




pertub = Tb_subset

