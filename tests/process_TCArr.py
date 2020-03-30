#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:01:02 2020

@author: benjamin
"""


import pandas as pd
import numpy as np
import scipy as sci

# load TC array

Arr1 = pd.read_csv("/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/TCArray/Arr1/TOA5_5134.TC_Array.dat", low_memory=False, skiprows=1)


Arr1.head()
Arr1.columns
Arr1_dropped =Arr1.drop(["RECORD",'BattV', 'PTemp_C','nmea_sentence(1)','nmea_sentence(2)'],axis=1)

Arr1_dropped = Arr1_dropped.drop([0,1])

TC1 = Arr1_dropped["Temp_C(1)"]
TC6 =  Arr1_dropped["Temp_C(6)"]
TC6 = TC6[20:]

np.correlate(float(TC1),TC6)