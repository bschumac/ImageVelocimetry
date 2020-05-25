#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:50:35 2020

@author: Benjamin Schumacher

Example file for correction of sonic anemometer data


"""




from sonic_func import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

irg2 = pd.read_csv("/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/IRG2/TOA5_irg_02_data0_cleaned.dat", header=[0,1], na_values='NAN')

irg1 = pd.read_csv("/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/IRG1/TOA5_irg_01_data2_cleaned.dat", header=[0,1], na_values='NAN')




def run_planarfit(dataframe, start_index):

    timestamp = dataframe["TIMESTAMP"].values[start_index:]
    u = dataframe["Ux"].values[start_index:]
    v = dataframe["Uy"].values[start_index:]
    w = dataframe["Uz"].values[start_index:]
    Ts = dataframe["Ts"].values[start_index:]
    
    timestamp_pf, u1_pf, v1_pf, w1_pf = planar_fit(u, v, w, sub_size = 10,timestamp=timestamp)
    Ts = Ts[:len(u1_pf)]
    return(timestamp_pf, u1_pf, v1_pf, w1_pf, Ts)



timestamp_irg02_pf, u1_irg02_pf, v1_irg02_pf, w1_irg02_pf, Ts_irg02 = run_planarfit(irg2, start_index=3800)

timestamp_irg01_pf, u1_irg01_pf, v1_irg01_pf, w1_irg01_pf, Ts_irg01 = run_planarfit(irg1, start_index=1300)










irg02_pf = pd.DataFrame({'TIMESTAMP': timestamp_irg02_pf.flatten(), 'Ux': u1_irg02_pf, 'Uy': v1_irg02_pf, 'Uz': w1_irg02_pf, 'Ts': Ts_irg02.flatten()})

irg01_pf = pd.DataFrame({'TIMESTAMP': timestamp_irg01_pf.flatten(), 'Ux': u1_irg01_pf, 'Uy': v1_irg01_pf, 'Uz': w1_irg01_pf, 'Ts': Ts_irg01.flatten()})

#example_rot3 = pd.DataFrame({'TIMESTAMP': timestamp_rot3.flatten(), 'Ux': u1_rot3, 'Uy': v1_rot3, 'Uz': w1_rot3})


irg02_pf.to_csv("/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/IRG2/TOA5_irg_02_tiltcorr_pf.csv")

irg01_pf.to_csv("/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/IRG1/TOA5_irg_01_tiltcorr_pf.csv")





#example_rot3.to_csv("/home/benjamin/Met_ParametersTST/GIT_code/sonicfun/example_result_triplerotation.csv")

#timestamp_rot3, u1_rot3, v1_rot3, w1_rot3 = triple_rot(u, v, w, sub_size = 10,timestamp=timestamp)