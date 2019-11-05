from functions.openpiv_fun import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import colors, cm
from joblib import Parallel, delayed
import multiprocessing
from functions.TST_fun import *
import datetime
from math import sqrt
import copy
import scipy as sio
import scipy.stats as stats


datapath = "/home/benjamin/Met_ParametersTST/PALMS/data/"
outpath = "/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test2/"
begin_value_images = 3540
end_value_images = begin_value_images+1
my_dpi = 300
#case = "strong_wind"
#case = "no_wind"
#case = "vertex_shedding"

lst_ws = [8,12,16,24,36,48]
lst_overlap = [[7,6,5,4],[11,10,9,8,7,6],[15,14,13,12,11,10],[23,22,21,20,19,18],
               [35,34,33,32,31,30,29,28,27,26,25,24,18],[47,45,41,40,36,32,30,24]]

lst_sa = [[12,16,24,32],[18,24,32,36],[20,24,32,40,48],[30,36,48,56,72],[42,54,72],[64]] 

case_lst = ["strong_wind","no_wind","vertex_shedding"]
multiplyer_lst = [1,2,3,4,6,8]
result_lst_names = ["case", "window_size", "overlap", "search_area_size", "ccussim", "ccvssim", "ccurmse", "ccvrmse",
                    "gsussim", "gsvssim", "gsurmse", "gsvrmse", "rmseussim", "rmsevssim", "rmseurmse", "rmsevrmse", 
                    "ssimussim", "ssimvssim", "ssimurmse", "ssimvrmse"]
                
result_lst = []
# case, ws, ol, sa, 

#case = case_lst[2]

case = "no_wind"

for case in case_lst:
    case = "no_wind"
    if case == "strong_wind":
        file = h5py.File(datapath+'cbl_surf.nc','r')
    elif case == "no_wind": 
        file = h5py.File(datapath+'cbl_nowind.nc','r')
    elif case == "vertex_shedding":
        file = h5py.File(datapath+'vort_large.nc','r')
    
    for i in range(0,len(lst_ws)):
        window_size = lst_ws[i]
        for j in range(0,len(lst_overlap[i])):
            act_result_lst = []
            overlap = lst_overlap[i][j]
            for k in range(0,len(lst_sa[i])):
                search_area_size = lst_sa[i][k]

                
                act_result_lst.append([case,window_size,overlap,search_area_size])
          
                window_size = 16
                overlap = 15
                search_area_size = 32
                overlap_search_area = 16


                u_model = file.get("u_xy")
                v_model = file.get("v_xy")
                
                
                
                if case == "strong_wind":
                       
                    u_model = np.reshape(u_model,(3600,256,512))
                    v_model = np.reshape(v_model,(3600,256,512))
                
                elif case == "no_wind":
                       
                    u_model = np.reshape(u_model,(3600,256,512))
                    v_model = np.reshape(v_model,(3600,256,512))
                    
                        
                elif case == "vertex_shedding": 
                    u_model = np.reshape(u_model,(3600,160,640))
                    v_model = np.reshape(v_model,(3600,160,640))
                
                u_subset = u_model[begin_value_images:end_value_images+1]
                v_subset = v_model[begin_value_images:end_value_images+1]
                
                windspeed = copy.copy(u_subset)
                
                for l in range(0,len(u_subset)):
                    windspeed[l] = calcwindspeed(u_subset[l],v_subset[l])    
                
               


                frame_a =  windspeed[0] 
                frame_b =  windspeed[1]  
                plt.figure()
                plt.imshow(frame_a)
                plt.colorbar()
    
                #plt.savefig(outpath+"cross_correlation/abca"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                #plt.close()
                u_model_cut = get_org_data(frame_a=u_model[begin_value_images], search_area_size=search_area_size, overlap=overlap )
                v_model_cut = get_org_data(frame_a=v_model[begin_value_images], search_area_size=search_area_size, overlap=overlap )

                
                # cross correlation
                
                try:
                    u, v = cross_correlation_aiv(frame_a, frame_b, window_size=window_size, overlap=overlap,
                                              dt=1, search_area_size=search_area_size, nfftx=None, 
                                              nffty=None,width=2, corr_method='fft', subpixel_method='gaussian')
                    
                    
                    x, y = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
    
                    ## vectorplot
                    u = np.flipud(u)
                    v = np.flipud(v)
                    plt.figure()
                    plt.imshow(frame_a)
                    plt.quiver(x,y,u,v)
                    #plt.show()
                    plt.savefig(outpath+"cross_correlation/IMAGEVELOCIMETRY_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()

                    
                    
                    
                    # histogram with measures
                    ssimvalcclstu = []
                    rmsevalcclstu = []
                    ssimvalcclstv = []
                    rmsevalcclstv = []
                    for multiplyer in multiplyer_lst:
                        
                        ssimvalcclstu.append(ssim(u*multiplyer,u_model_cut, axis= (0,1)))
                        rmsevalcclstu.append(np.sqrt(np.mean((u_model_cut-u*multiplyer)**2,(0,1))))
                        ssimvalcclstv.append(ssim(v*multiplyer,v_model_cut, axis= (0,1)))
                        rmsevalcclstv.append(np.sqrt(np.mean((v_model_cut-v*multiplyer)**2,(0,1))))
                    
                    ssimvalccu = np.max(ssimvalcclstu)
                    ssimvalccv = np.max(ssimvalcclstv)
                    rmsevalccu = np.min(rmsevalcclstu)
                    rmsevalccv = np.min(rmsevalcclstv)
                    
                    ssimidxu = ssimvalcclstu.index(np.max(ssimvalccu))
                    ssimidxv = ssimvalcclstv.index(np.max(ssimvalccv))
                    rmseidxu = rmsevalcclstu.index(np.min(rmsevalccu))
                    rmseidxv = rmsevalcclstv.index(np.min(rmsevalccv))
                    
                    u[numpy.isnan(u)]=0
                    v[numpy.isnan(v)]=0
                    u_model_cut = np.flipud(u_model_cut)
                    v_model_cut = np.flipud(v_model_cut)
                    plt.figure()
                    plt.imshow(frame_a)
                    plt.colorbar()
                    plt.quiver(x,y,u_model_cut,v_model_cut)
                    #plt.show()
                    plt.savefig(outpath+"cross_correlation/MODEL_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    
                    dif_u = u*multiplyer_lst[ssimidxu]-u_model_cut
                    dif_flatten = dif_u.flatten()[~numpy.isnan(dif_u.flatten())]
                    if (np.abs(dif_flatten) > 15).all():
                        breaks = np.linspace(np.min(dif_flatten), np.max(dif_flatten), 50)
                    else:
                        breaks = np.linspace(-15, 15, 50)
                    n, x, _= plt.hist(dif_flatten, bins=breaks, histtype=u'step', density=True)
                    density = stats.gaussian_kde(dif_flatten)
                    plt.text(-1,np.max(n)/2, s= "RMSE: "+str(round(rmsevalccu,2)) )
                    plt.text(-1,(np.max(n)/2)-(np.max(n)/4), s= "SSIM: "+str(round(ssimvalccu,2)) )
                    plt.title("U component multiplyer: "+str(multiplyer_lst[ssimidxu]))
                    plt.plot(x, density(x))
                    #plt.show()
                    plt.savefig(outpath+"cross_correlation/hist_u_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    
                    
                    dif_u = v*multiplyer_lst[ssimidxv]-u_model_cut
                    dif_flatten = dif_u.flatten()[~numpy.isnan(dif_u.flatten())]
                    if (np.abs(dif_flatten) > 15).all():
                        breaks = np.linspace(np.min(dif_flatten), np.max(dif_flatten), 50)
                    else:
                        breaks = np.linspace(-15, 15, 50)
                    n, x, _= plt.hist(dif_flatten, bins=breaks, histtype=u'step', density=True)
                    density = stats.gaussian_kde(dif_flatten)
                    plt.text(-1,np.max(n)/2, s= "RMSE: "+str(round(rmsevalccv,2)) )
                    plt.text(-1,(np.max(n)/2)-(np.max(n)/4), s= "SSIM: "+str(round(ssimvalccv,2)) )
                    plt.title("V component multiplyer: "+str(multiplyer_lst[ssimidxv]))
                    plt.plot(x, density(x))
                    #plt.show()
                    plt.savefig(outpath+"cross_correlation/hist_v_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    act_result_lst.append([ssimvalccu,ssimvalccv,rmsevalccu,rmsevalccv])   
                except:
                    pass
                
                # greyscale
                
                try:
                    
                    
                    u1, v1= window_correlation_tiv(frame_a, frame_b, window_size_x=window_size, overlap_window=overlap, overlap_search_area=overlap_search_area, corr_method="greyscale", search_area_size_x=search_area_size, 
                           search_area_size_y=0,  window_size_y=0, mean_analysis = False, std_analysis = False, std_threshold = 10)
                    
                    
                    x1, y1 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
                    
                    
                    u1=np.flipud(u1)
                    v1=np.flipud(v1)
                    #v1 = v1 *-1
                    #u1 = u1*-1
                    #u1[numpy.isnan(u1)]=0
                    #v1[numpy.isnan(v1)]=0
                    #plt.imshow(u1)
                    #plt.imshow(v1)
                    #plt.imshow(v_model_cut)
                    
                    plt.figure(1)
                    #plt.subplot(211)
                    #plt.gca().set_title("IV")
                    plt.imshow(frame_a,vmin=0,vmax=7)
                    plt.colorbar()
                    plt.quiver(x1,y1,u1,v1,color="white")
                    #streamplot(U=u1, V=v1*-1, X=x1, Y=y1, topo = frame_a,vmin=0,vmax=8)
                    #plt.show()
                    plt.savefig(outpath+"greyscale/abcIMAGEVELOCIMETRY_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=1200,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    lw = .3
                    plt.figure()

                    plt.imshow(frame_a,vmin=0,vmax=7)
                    plt.colorbar()
                    
                    plt.quiver(x1,y1,np.flipud(u_model_cut),np.flipud(v_model_cut), color='white', linewidth=lw)
                    #streamplot(U=np.flipud(u_model_cut), V=np.flipud(v_model_cut)*-1, X=x1, Y=y1, topo = frame_a,vmin=0,vmax=8)
                    
                    plt.savefig(outpath+"greyscale/abcMODEL_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=1200,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()    
                    
                    
                    
                    
                    # histogram with measures
                    ssimvalcclstu = []
                    rmsevalcclstu = []
                    ssimvalcclstv = []
                    rmsevalcclstv = []
                    for multiplyer in multiplyer_lst:
                        
                        ssimvalcclstu.append(ssim(u1*multiplyer,u_model_cut, axis= (0,1)))
                        rmsevalcclstu.append(np.sqrt(np.mean((u_model_cut-u1*multiplyer)**2,(0,1))))
                        ssimvalcclstv.append(ssim(v1*multiplyer,v_model_cut, axis= (0,1)))
                        rmsevalcclstv.append(np.sqrt(np.mean((v_model_cut-v1*multiplyer)**2,(0,1))))
                    
                    ssimvalccu = np.max(ssimvalcclstu)
                    ssimvalccv = np.max(ssimvalcclstv)
                    rmsevalccu = np.min(rmsevalcclstu)
                    rmsevalccv = np.min(rmsevalcclstv)
                    
                    ssimidxu = ssimvalcclstu.index(np.max(ssimvalccu))
                    ssimidxv = ssimvalcclstv.index(np.max(ssimvalccv))
                    rmseidxu = rmsevalcclstu.index(np.min(rmsevalccu))
                    rmseidxv = rmsevalcclstv.index(np.min(rmsevalccv))
                    
                    
                                   
                    
                    
                    dif_u = u1*multiplyer_lst[ssimidxu]-u_model_cut
                    dif_flatten = dif_u.flatten()[~numpy.isnan(dif_u.flatten())]
                    if (np.abs(dif_flatten) > 15).all():
                        breaks = np.linspace(np.min(dif_flatten), np.max(dif_flatten), 50)
                    else:
                        breaks = np.linspace(-15, 15, 50)
                    n, x, _= plt.hist(dif_flatten, bins=breaks, histtype=u'step', density=True)
                    density = stats.gaussian_kde(dif_flatten)
                    plt.text(-1,np.max(n)/2, s= "RMSE: "+str(round(rmsevalccu,2)) )
                    plt.text(-1,(np.max(n)/2)-(np.max(n)/4), s= "SSIM: "+str(round(ssimvalccu,2)) )
                    plt.title("U component multiplyer: "+str(multiplyer_lst[ssimidxu]))
                    plt.plot(x, density(x))
                    #plt.show()
                    plt.savefig(outpath+"greyscale/hist_u_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    
                    
                    
                    dif_u = v1*multiplyer_lst[ssimidxv]-u_model_cut
                    dif_flatten = dif_u.flatten()[~numpy.isnan(dif_u.flatten())]
                    if (np.abs(dif_flatten) > 15).all():
                        breaks = np.linspace(np.min(dif_flatten), np.max(dif_flatten), 50)
                    else:
                        breaks = np.linspace(-15, 15, 50)
                    n, x, _= plt.hist(dif_flatten, bins=breaks, histtype=u'step', density=True)
                    density = stats.gaussian_kde(dif_flatten)
                    plt.text(-1,np.max(n)/2, s= "RMSE: "+str(round(rmsevalccv,2)) )
                    plt.text(-1,(np.max(n)/2)-(np.max(n)/4), s= "SSIM: "+str(round(ssimvalccv,2)) )
                    plt.title("V component multiplyer: "+str(multiplyer_lst[ssimidxv]))
                    plt.plot(x, density(x))
                    #plt.show()
                    plt.savefig(outpath+"greyscale/hist_v_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    act_result_lst.append([ssimvalccu,ssimvalccv,rmsevalccu,rmsevalccv])
                
                
                
                # rmse
                
                
                
                    u2, v2= window_correlation_tiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                                           search_area_size_x=search_area_size, search_area_size_y=0, corr_method="rmse")
                    
                    
                    x2, y2 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
                    
                    
                    #v2 = v2 *-1
                    
                    #u1=np.flip(u1,0)
                    #v1=np.flip(v1,0)
                    u2[numpy.isnan(u2)]=0
                    v2[numpy.isnan(v2)]=0
                    
                    plt.figure()
                    plt.imshow(frame_a)
                    plt.colorbar()
                    plt.quiver(x2,y2,u2,v2)
                    #plt.show()
                    plt.savefig(outpath+"rmse/IMAGEVELOCIMETRY_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    
                    # histogram with measures
                    ssimvalcclstu = []
                    rmsevalcclstu = []
                    ssimvalcclstv = []
                    rmsevalcclstv = []
                    for multiplyer in multiplyer_lst:
                        
                        ssimvalcclstu.append(ssim(u2*multiplyer,u_model_cut, axis= (0,1)))
                        rmsevalcclstu.append(np.sqrt(np.mean((u_model_cut-u2*multiplyer)**2,(0,1))))
                        ssimvalcclstv.append(ssim(v2*multiplyer,v_model_cut, axis= (0,1)))
                        rmsevalcclstv.append(np.sqrt(np.mean((v_model_cut-v2*multiplyer)**2,(0,1))))
                    
                    ssimvalccu = np.max(ssimvalcclstu)
                    ssimvalccv = np.max(ssimvalcclstv)
                    rmsevalccu = np.min(rmsevalcclstu)
                    rmsevalccv = np.min(rmsevalcclstv)
                    
                    ssimidxu = ssimvalcclstu.index(np.max(ssimvalccu))
                    ssimidxv = ssimvalcclstv.index(np.max(ssimvalccv))
                    rmseidxu = rmsevalcclstu.index(np.min(rmsevalccu))
                    rmseidxv = rmsevalcclstv.index(np.min(rmsevalccv))
                    
                    
                    
                    dif_u = u2*multiplyer_lst[ssimidxu]-u_model_cut
                    dif_flatten = dif_u.flatten()[~numpy.isnan(dif_u.flatten())]
                    if (np.abs(dif_flatten) > 6).all():
                        breaks = np.linspace(np.min(dif_flatten), np.max(dif_flatten), 50)
                    else:
                        breaks = np.linspace(-15, 15, 50)
                    n, x, _= plt.hist(dif_flatten, bins=breaks, histtype=u'step', density=True)
                    density = stats.gaussian_kde(dif_flatten)
                    plt.text(-1,np.max(n)/2, s= "RMSE: "+str(round(rmsevalccu,2)) )
                    plt.text(-1,(np.max(n)/2)-(np.max(n)/4), s= "SSIM: "+str(round(ssimvalccu,2)) )
                    plt.title("U component multiplyer: "+str(multiplyer_lst[ssimidxu]))
                    plt.plot(x, density(x))
                    #plt.show()
                    plt.savefig(outpath+"rmse/hist_u_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    plt.figure()
                    plt.imshow(frame_a)
                    plt.colorbar()
                    plt.quiver(x2,y2,u_model_cut,v_model_cut)
                    #plt.show()
                    plt.savefig(outpath+"greyscale/MODEL_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    
                    dif_u = v2*multiplyer_lst[ssimidxv]-u_model_cut
                    dif_flatten = dif_u.flatten()[~numpy.isnan(dif_u.flatten())]
                    if (np.abs(dif_flatten) > 15).all():
                        breaks = np.linspace(np.min(dif_flatten), np.max(dif_flatten), 50)
                    else:
                        breaks = np.linspace(-15, 15, 50)
                    n, x, _= plt.hist(dif_flatten, bins=breaks, histtype=u'step', density=True)
                    density = stats.gaussian_kde(dif_flatten)
                    plt.text(-1,np.max(n)/2, s= "RMSE: "+str(round(rmsevalccv,2)) )
                    plt.text(-1,(np.max(n)/2)-(np.max(n)/4), s= "SSIM: "+str(round(ssimvalccv,2)) )
                    plt.title("V component multiplyer: "+str(multiplyer_lst[ssimidxv]))
                    plt.plot(x, density(x))
                    #plt.show()
                    plt.savefig(outpath+"rmse/hist_v_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    act_result_lst.append([ssimvalccu,ssimvalccv,rmsevalccu,rmsevalccv])
                
                except:
                    pass
                try:
                # ssim
                
                
                    u3, v3= window_correlation_tiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
                                           search_area_size_x=search_area_size, search_area_size_y=0, corr_method="ssim")
                    
                    
                    x3, y3 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
                    
                    #u1=np.flip(u1,0)
                    #v1=np.flip(v1,0)
                    #v3 = v3 *-1
                    
                    plt.figure()
                    plt.imshow(frame_a)
                    plt.colorbar()
                    plt.quiver(x3,y3,u3,v3)
                    #plt.show()
                    plt.savefig(outpath+"ssim/IMAGEVELOCIMETRY_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    u3[numpy.isnan(u3)]=0
                    v3[numpy.isnan(v3)]=0
                    
                    # histogram with measures
                    ssimvalcclstu = []
                    rmsevalcclstu = []
                    ssimvalcclstv = []
                    rmsevalcclstv = []
                    for multiplyer in multiplyer_lst:
                        
                        ssimvalcclstu.append(ssim(u3*multiplyer,u_model_cut, axis= (0,1)))
                        rmsevalcclstu.append(np.sqrt(np.mean((u_model_cut-u3*multiplyer)**2,(0,1))))
                        ssimvalcclstv.append(ssim(v3*multiplyer,v_model_cut, axis= (0,1)))
                        rmsevalcclstv.append(np.sqrt(np.mean((v_model_cut-v3*multiplyer)**2,(0,1))))
                    
                    ssimvalccu = np.max(ssimvalcclstu)
                    ssimvalccv = np.max(ssimvalcclstv)
                    rmsevalccu = np.min(rmsevalcclstu)
                    rmsevalccv = np.min(rmsevalcclstv)
                    
                    ssimidxu = ssimvalcclstu.index(np.max(ssimvalccu))
                    ssimidxv = ssimvalcclstv.index(np.max(ssimvalccv))
                    rmseidxu = rmsevalcclstu.index(np.min(rmsevalccu))
                    rmseidxv = rmsevalcclstv.index(np.min(rmsevalccv))
                    
                    
                    
                    plt.figure()
                    plt.imshow(frame_a)
                    plt.colorbar()
                    plt.quiver(x3,y3,u_model_cut,v_model_cut)
                    plt.show()
                    plt.savefig(outpath+"ssim/MODEL_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    dif_u = u3*multiplyer_lst[ssimidxu]-u_model_cut
                    dif_flatten = dif_u.flatten()[~numpy.isnan(dif_u.flatten())]
                    if (np.abs(dif_flatten) > 15).all():
                        breaks = np.linspace(np.min(dif_flatten), np.max(dif_flatten), 50)
                    else:
                        breaks = np.linspace(-15, 15, 50)
                    n, x, _= plt.hist(dif_flatten, bins=breaks, histtype=u'step', density=True)
                    density = stats.gaussian_kde(dif_flatten)
                    plt.text(-1,np.max(n)/2, s= "RMSE: "+str(round(rmsevalccu,2)) )
                    plt.text(-1,(np.max(n)/2)-(np.max(n)/4), s= "SSIM: "+str(round(ssimvalccu,2)) )
                    plt.title("U component multiplyer: "+str(multiplyer_lst[ssimidxu]))
                    plt.plot(x, density(x))
                    #plt.show()
                    plt.savefig(outpath+"ssim/hist_u_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    
                    
                    
                    dif_u = v3*multiplyer_lst[ssimidxv]-u_model_cut
                    dif_flatten = dif_u.flatten()[~numpy.isnan(dif_u.flatten())]
                    if (np.abs(dif_flatten) > 15).all():
                        breaks = np.linspace(np.min(dif_flatten), np.max(dif_flatten), 50)
                    else:
                        breaks = np.linspace(-15, 15, 50)
                    n, x, _= plt.hist(dif_flatten, bins=breaks, histtype=u'step', density=True)
                    density = stats.gaussian_kde(dif_flatten)
                    plt.text(-1,np.max(n)/2, s= "RMSE: "+str(round(rmsevalccv,2)) )
                    plt.text(-1,(np.max(n)/2)-(np.max(n)/4), s= "SSIM: "+str(round(ssimvalccv,2)) )
                    plt.title("V component multiplyer: "+str(multiplyer_lst[ssimidxv]))
                    plt.plot(x, density(x))
                    #plt.show()
                    plt.savefig(outpath+"ssim/hist_v_case_"+str(case)+"_WS_"+str(window_size)+"_OL_"+str(overlap)+"_SA_"+str(search_area_size)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
                    plt.close()
                    act_result_lst.append([ssimvalccu,ssimvalccv,rmsevalccu,rmsevalccv])
                except:
                    pass
                
                result_lst.append(act_result_lst)
                
                
                
                

with open('/home/benjamin/Met_ParametersTST/PALMS/sensitivity_test2/sensitivity_test_performance.txt', 'w') as f:
    str_act_entry = str(result_lst_names)
    str_act_entry = str_act_entry.replace("]","")
    str_act_entry = str_act_entry.replace("[","")
    str_act_entry = str_act_entry.replace("'","")
    f.write("%s\n" % str_act_entry)
    act_entry= []
    for i in range(0, len(result_lst)):
        actlst = result_lst[i]
        act_entry = []
        
        for j in range(0,len(actlst)):
            if j != 0 and type(actlst[j][0]) is str:
                str_act_entry = str(act_entry)
                str_act_entry = str_act_entry.replace("]","")
                str_act_entry = str_act_entry.replace("[","")
                str_act_entry = str_act_entry.replace("'","")
                
                f.write("%s\n" % str_act_entry)
                act_entry= []
                act_entry = act_entry+actlst[j]
            else:    
                act_entry = act_entry+actlst[j]
    
 
### old stuff
#
#
#for i in range(begin_value_images-120,end_value_images):
#   
#    frame_a = windspeed[i]
#    frame_b = windspeed[i+1]
#    
#    u, v = cross_correlation_aiv(frame_a, frame_b, window_size=window_size, overlap=overlap,
#                              dt=1, search_area_size=search_area_size, nfftx=None, 
#                              nffty=None,width=2, corr_method='fft', subpixel_method='gaussian')
#    
#    
#    x, y = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
#
#    u1=np.flip(u1,0)
#    v1=np.flip(v1,0)
#    
#    plt.figure()
#    plt.imshow(frame_a)
#    plt.quiver(x,y,u,v)
#    plt.savefig(datapath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
#    plt.close()
#
#datapath = "/home/benjamin/Met_ParametersTST/PALMS/test_iv_greyscale/"
#
#
#
#for i in range(begin_value_images-120,end_value_images):
#    #i = begin_value_images   
#    frame_a = windspeed[i]
#    frame_b = windspeed[i+1]
#
#
#    u1, v1= window_correlation_aiv(frame_a, frame_b, window_size_x=window_size, window_size_y=0, overlap=overlap, 
#                           search_area_size_x=search_area_size, search_area_size_y=0, corr_method="ssim")
#    
#    
#    x1, y1 = get_coordinates( image_size=frame_a.shape, window_size=search_area_size, overlap=overlap )
#    
#    
#    plt.figure()
#    plt.imshow(frame_a)
#    plt.quiver(x1,y1,u1,v1)
#    #plt.show()
#    
#    plt.savefig(datapath+str(i)+".png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
#    plt.close()
#
#
#
#
#### kmeans
#
#kmeans_test = tst_kmeans(dataset=windspeed_subset,n_clusters = 6)
#plt.imshow(kmeans_test[1])
#plt.colorbar()
#from skimage import measure
#
## Find contours at a constant value of 0.8
#contours = measure.find_contours(kmeans_test[1], 0)
#
## Display the image and plot all contours found
#ax.imshow(kmeans_test[1], interpolation='nearest', cmap=plt.cm.gray)
#
#
#for n, contour in enumerate(contours):
#    if n>4 and n <= 7:
#        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
#
#
#
#
