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
    file = h5py.File(datapath+'Tb_stab_pertub_py_virdris.nc','r')



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

corrmap = np.zeros((pertub.shape[1],pertub.shape[2]))

for i in range(0,pertub.shape[1]):
    print(i)
    for j in range(0,pertub.shape[2]):
        corrmap[i,j] = np.corrcoef(minmean(irg2_subset.iloc[:,3],10),pertub[21:,i,j])[0,1]

plt.imshow(pertub[45])

my_dpi = 1200
outpath="/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/corrmaps/"
plt.imshow(corrmap)
plt.colorbar()
plt.savefig(outpath+"w_corrmap.png",dpi=my_dpi,bbox_inches='tight',pad_inches = 0,transparent=False)
plt.close()



corrmap.max()





