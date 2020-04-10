#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:58:13 2019

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
from functions.TST_fun import *
from scipy import statss
import datetime
from math import sqrt
#import openpiv.filters
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


u_minmean_irg2 = u_minmean_irg2 = minmean(irg2_subset.iloc[:,1], 20) 
v_minmean_irg2 = minmean(irg2_subset.iloc[:,2], 20) 


dates = list(irg2_subset.index)
list(irg2_subset.columns) 
unique_dates = []
prev_date = ''
for date in dates:
    act_date = date[0:19]
    unique_dates.append(act_date)
irg2_subset['substr_date'] = pd.to_datetime(unique_dates)       
len(unique_dates)
irg2_subset.head()


irg2_subset_sec = irg2_subset.groupby([irg2_subset['substr_date'].dt.time])['u3', 'v3', 'w3'].mean()    
irg2_subset_sec.to_csv('/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/IRG2/TOA5_irg_02_tiltcorr_meansec.csv')
irg2_subset.resample('u3', on='substr_date').mean()
wd_minmean_irg2 = calcwinddirection(u_minmean_irg2, v_minmean_irg2)
