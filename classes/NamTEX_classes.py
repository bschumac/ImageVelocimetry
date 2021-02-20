#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 03:07:36 2021

@author: benjamin
"""
import sys
import socket
hostname = (socket.gethostname())
import pandas as pd
import numpy as np
import datetime
import itertools
import copy
from scipy import fftpack
from scipy import signal
import pandas as pd

if hostname == "PHOBOS":
    sys.path.insert(1, '/home/benjamin/Met_ParametersTST/GIT_code/code/functions/')
    sys.path.insert(1, '/home/benjamin/Met_ParametersTST/GIT_code/code/classes/')
else:   
    sys.path.insert(1, '/home/rccuser/.jupyter/code/ImageVelocimetry/functions/')

from TST_fun import *    
    
    
class sonictower:
    def __init__(self, pathlst = [], soniclst = [], sonicheights = [], azimuth = 0):
        self.pathlst = pathlst
        self.soniclst = soniclst
        self.sonicheights = sonicheights
        self.azimuth = azimuth
        print("Read Sonics...")
        self.read_soniclst()
        
        print("Tilting sonics to azimuth")
        self.tilt_sonics()
    
    def read_soniclst(self):
        for path in self.pathlst:
            self.soniclst.append(pd.read_csv(path))
    
    def tilt_sonics(self):
        if len(self.sonicheights) == 0:
            print("Please define sonic heights!")
        elif len(self.sonicheights) != len(self.soniclst):
            print("More heights than sonic dataframes")
        else:
            for i in range(0,len(self.sonicheights)):
                new_v = (self.soniclst[i]['Ux'].values*np.cos(self.azimuth))+ (self.soniclst[i]['Uy'].values*np.sin(self.azimuth))
                new_u = -self.soniclst[i]['Ux'].values*np.sin(self.azimuth)+np.cos(self.azimuth)*self.soniclst[i]['Uy'].values
                self.soniclst[i]['u_tilt'] = new_u 
                self.soniclst[i]['v_tilt'] = new_v
                
    
     

class experiment(sonictower):
    
    def __init__(self, exp_name, exp_num, start_time, end_time, ass_uas, ass_vas, tb_org_stab, var_dict, sonictower, sonic_log_f):
        self.exp_name = exp_name
        self.exp_num = exp_num
        
    
        self.start_time = start_time
        self.end_time = end_time
        super().__init__(pathlst = [], soniclst = sonictower.soniclst, sonicheights = sonictower.sonicheights, azimuth = sonictower.azimuth)
        self.num_sonics = len(self.soniclst)
        self.sonic_log_f = sonic_log_f
        self.sonics_trimmed = []
        
        
        # A-TIV
        self.ass_uas = ass_uas
        self.ass_vas = ass_vas
        self.ass_ws = calcwindspeed(ass_vas, ass_uas)
        self.tb_org_stab = tb_org_stab
        self.var_dict = var_dict
        
        self.tb_subsamp = None
        self.tb_perturb60 = None
        self.var_dict = var_dict
        
        # if for larger size FOV
        
        if self.ass_uas.shape[1] < 200:
            self.uas_spmean_l = np.nanmean(self.ass_uas[:,40:130,75:165],axis=(1,2)) 
            self.vas_spmean_l = np.nanmean(self.ass_vas[:,40:130,75:165],axis=(1,2))
            self.ws_spmean_l = np.nanmean(self.ass_ws[:,40:130,75:165],axis=(1,2))

            self.uas_spmean_m = np.nanmean(self.ass_uas[:,60:110,95:145],axis=(1,2)) 
            self.vas_spmean_m = np.nanmean(self.ass_vas[:,60:110,95:145],axis=(1,2))
            self.ws_spmean_m = np.nanmean(self.ass_ws[:,60:110,95:145],axis=(1,2))
        else:
            self.uas_spmean_l = np.nanmean(self.ass_uas[:,80:320,130:390],axis=(1,2)) 
            self.vas_spmean_l = np.nanmean(self.ass_vas[:,80:320,130:390],axis=(1,2))
            self.ws_spmean_l = np.nanmean(self.ass_ws[:,80:320,130:390],axis=(1,2))

            self.uas_spmean_m = np.nanmean(self.ass_uas[:,120:280,180:340],axis=(1,2)) 
            self.vas_spmean_m = np.nanmean(self.ass_vas[:,120:280,180:340],axis=(1,2))
            self.ws_spmean_m = np.nanmean(self.ass_ws[:,120:280,180:340],axis=(1,2))
            
        self.ativ_means = {"uas_spmean_l":self.uas_spmean_l, "vas_spmean_l":self.vas_spmean_l, "ws_spmean_l": self.ws_spmean_l, 
                           "uas_spmean_m":self.uas_spmean_m, "vas_spmean_m":self.vas_spmean_m, "ws_spmean_m":self.ws_spmean_m}

        
        # sonic processed
        self.soniclst_sec = [] 
        
        self.u_coef_lst = []
        self.v_coef_lst = []
        self.ws_coef_lst = [] 
        
        print("Trimming sonics to exp time..")
        self.trim_sonictime()
        print("Group sonics to 1Hz")
        self.group_sonics()
    
    
  
    def trim_sonictime(self):
         
        if len(self.sonicheights) == 0:
            print("Please define sonic heights!")
        elif len(self.sonicheights) != len(self.soniclst):
            print("More heights than sonic dataframes")
        else:
            for i in range(0,len(self.sonicheights)):
                self.soniclst[i]['TIMESTAMP'] = self.soniclst[i]['TIMESTAMP'].astype(str)
                self.soniclst[i]['TIMESTAMP']= pd.to_datetime(self.soniclst[i]['TIMESTAMP'],format='%Y-%m-%d %H:%M:%S.%f')
                self.soniclst[i] = self.soniclst[i].set_index(pd.DatetimeIndex(self.soniclst[i]['TIMESTAMP'].values[:].flatten()))
                d0 = datetime.datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S.%f')
                d1= datetime.datetime.strptime(self.end_time, '%Y-%m-%d %H:%M:%S.%f')
                
                
                self.sonics_trimmed.append(self.soniclst[i].loc[np.datetime64(d0):np.datetime64(d1)]) 
  
  
    def group_sonics(self):
       
        #### calc groups ####

        lst = range(0,int(len(self.sonics_trimmed[0])/(self.sonic_log_f+1)))
        print(int(len(self.sonics_trimmed[0])/(self.sonic_log_f+1)))
        groups = list(itertools.chain.from_iterable(itertools.repeat(x, self.sonic_log_f+1) for x in lst))
        if len(groups) > len(self.sonics_trimmed[0]):
            groups = groups[:len(self.sonics_trimmed[0])-len(groups)]
        elif len(groups) < len(self.sonics_trimmed[0]): 
            last_idx = groups[-1]
            for i in range(len(self.sonics_trimmed[0])- len(groups)):
                groups.append(last_idx+1)
        for i in range(0,len(self.sonics_trimmed)):
            self.sonics_trimmed[i]['groups'] = groups
            self.soniclst_sec.append(self.sonics_trimmed[i].groupby(self.sonics_trimmed[i]['groups']).mean())
        




    def plot_sonics(self):
        for sonic in self.soniclst_sec:
            plt.plot(sonic['u_tilt'].values[:])
        plt.title("U-windspeed")
        plt.show()
        for sonic in self.soniclst_sec:
            plt.plot(sonic['v_tilt'].values[:])
        plt.title("V-windspeed")
        plt.show()

        

    def calccoef(self):
        coef_df = []
        
        names_lst = ["u_coef_ativl",  "v_coef_ativl",  "ws_coef_ativl",  
                       "u_coef_ativm",  "v_coef_ativm",  "ws_coef_ativm"]
        var_lst = ['u_tilt', 'v_tilt', 'ws', 'u_tilt', 'v_tilt', 'ws']
        
        for k in range(0, len(self.sonicheights)):
            processing_dict = {} 
            j = -1
            #for j in range(0,len(self.ativ_means.values())):
            for key in self.ativ_means.keys():   
                j += 1
                if var_lst[j] == "ws":
                    irg_val = calcwindspeed(self.soniclst_sec[k][var_lst[0]], self.soniclst_sec[k][var_lst[1]])
                else:    
                    irg_val = self.soniclst_sec[k][var_lst[j]]
                        
                ativmean = self.ativ_means[key]

                out_coeflst =[]
                dif_len = np.abs(len(irg_val)-len(ativmean))
                if len(irg_val) > len(ativmean):
                    for i in range(0,dif_len):
                        coef = np.corrcoef(irg_val[i:len(irg_val)-dif_len+i],ativmean*-1)
                        out_coeflst.append(coef[0,1])
                elif len(irg_val) < len(ativmean):
                    for i in range(0,dif_len):
                        coef = np.corrcoef(irg_val,ativmean[i:len(ativmean)-dif_len+i]*-1)
                        out_coeflst.append(coef[0,1])
                else:
                    coef = np.corrcoef(irg_val,ativmean*-1)
                    out_coeflst.append(coef[0,1])
                
                if np.mean(out_coeflst) < 0:
                    processing_dict[names_lst[j]]=np.min(out_coeflst)*-1
                    processing_dict["len_coeflst_"+names_lst[j]] = len(out_coeflst)
                    processing_dict["arg_max_"+names_lst[j]] = np.argmin(out_coeflst)
                else:
                    processing_dict[names_lst[j]]=np.max(out_coeflst)
                    processing_dict["arg_max_"+names_lst[j]] = np.argmax(out_coeflst)
                    processing_dict["len_coeflst_"+names_lst[j]] = len(out_coeflst)
                
                processing_dict["Experiment_"+self.exp_name+"_num"] = self.exp_num    
                processing_dict[names_lst[j]+str("len")] = len(out_coeflst)
                processing_dict["sonicheight"] = str(self.sonicheights[k])
                processing_dict["Mean_Sonic_WS"] = np.mean(irg_val)
            coef_df.append(processing_dict)                  
            
        return(pd.DataFrame(coef_df))
    
    def plot_psd(self, height, sonic_varname, ativ_varname):
        sonic_num = self.sonicheights.index(height)
        
        if sonic_varname != "ws":
            sonic_val = self.soniclst_sec[sonic_num][sonic_varname] *-1            
            ativ_val = self.ativ_means[ativ_varname]
         
        else:
            sonic_val = calcwindspeed(self.soniclst_sec[sonic_num]["v_tilt"], self.soniclst_sec[sonic_num]["u_tilt"])
            ativ_val = self.ativ_means[ativ_varname]     
        
        f_s, Pxx_s = signal.welch(sonic_val, fs=0.9284, nperseg=256, detrend=False)
        f_ativ, Pxx_ativ = signal.welch(ativ_val, fs=0.915, nperseg=256, detrend=False)


        a = np.array([0.03,0.1,1])

        b = np.power(10,np.log10(a)*(-1*5/3)-0.75)


        plt.yscale("log")
        plt.xscale("log")
        plt.plot(f_s,Pxx_s)
        plt.plot(f_ativ,Pxx_ativ)
        plt.plot(a,b)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title("Sonic Height: "+str(height)+" m")
        
        
    def add_psdplot(self, height, sonic_varname, ativ_varname):
        sonic_num = self.sonicheights.index(height)
        
        if sonic_varname != "ws":
            sonic_val = self.soniclst_sec[sonic_num][sonic_varname] *-1            
            ativ_val = self.ativ_means[ativ_varname]
         
        else:
            sonic_val = calcwindspeed(self.soniclst_sec[sonic_num]["v_tilt"], self.soniclst_sec[sonic_num]["u_tilt"])
            ativ_val = self.ativ_means[ativ_varname]     
        
        f_s, Pxx_s = signal.welch(sonic_val, fs=0.9284, nperseg=256, detrend=False)
        f_ativ, Pxx_ativ = signal.welch(ativ_val, fs=0.915, nperseg=256, detrend=False)


        plt.yscale("log")
        plt.xscale("log")
        plt.plot(f_s,Pxx_s*f_s)
        plt.plot(f_ativ,Pxx_ativ*f_ativ)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title("Sonic Height: "+str(height)+" m")
        
        
    
    
    
    
    
    
    
    def plot_ps(self, height, sonic_varname, ativ_varname):
        
        sonic_num = self.sonicheights.index(height)
        
        if sonic_varname != "ws":
            sonic_val = self.soniclst_sec[sonic_num][sonic_varname] *-1            
            ativ_val = self.ativ_means[ativ_varname]
         
        else:
            sonic_val = calcwindspeed(self.soniclst_sec[sonic_num]["v_tilt"], self.soniclst_sec[sonic_num]["u_tilt"])
            ativ_val = self.ativ_means[ativ_varname]     
                
        sampling_rate=0.9284
        fourier_transform = np.fft.rfft(sonic_val *-1)
        abs_fourier_transform = np.abs(fourier_transform)
        power_spectrum = np.square(abs_fourier_transform)
        frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))
        plt.plot(frequency, power_spectrum, alpha=0.7)

        sampling_rate=0.915
        fourier_transform = np.fft.rfft(ativ_val)
        abs_fourier_transform = np.abs(fourier_transform)
        power_spectrum = np.square(abs_fourier_transform)
        frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))


        plt.plot(frequency, power_spectrum, alpha=0.7)
        plt.ylim(-1,500)
        plt.title("Sonic Height: "+str(height)+" m")



    
    
     
