#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:57:46 2019

@author: benjamin
"""

from matplotlib import pyplot as plt
import numpy as np
import h5py

from functions.TST_fun import create_tst_pertubations_mm


from scipy.stats import kurtosis, skew

file = h5py.File("/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Tb_stab_cut_red_20Hz.nc",'r')

tb20hz = file.get("Tb")

c = np.array(tb20hz)

tb2hz = tb20hz[1::10]



tb_2hz_pertub = create_tst_pertubations_mm(tb2hz)



tb_2hz_pertub_cut = tb_2hz_pertub[:,50:150,50:150]

tb_2hz_pertub_cut[1]


tb_2hz_pertub_cut
plt.imshow(tb_2hz_pertub_cut[1])
np.min(tb_2hz_pertub_cut)
breaks=np.arange(-4, 4, 0.05).tolist()


skewness_lst = []

for i in range(0,len(tb_2hz_pertub_cut)):

    act_skew = skew(tb_2hz_pertub_cut[i].reshape(tb_2hz_pertub_cut[i].shape[0]*tb_2hz_pertub_cut[i].shape[1]))
    skewness_lst.append(act_skew)

np.mean(skewness_lst)

plt.hist(tb_2hz_pertub_cut.reshape(tb_2hz_pertub_cut.shape[0]*tb_2hz_pertub_cut.shape[1]*tb_2hz_pertub_cut.shape[2]), bins=breaks)

skew(tb_2hz_pertub_cut.reshape(tb_2hz_pertub_cut.shape[0]*tb_2hz_pertub_cut.shape[1]*tb_2hz_pertub_cut.shape[2]))


datapath = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/"
out_fld_name = "streamplot/"
v_new_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/v_TIV_test_new_biggerIW_0-400.nc"
u_new_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/u_TIV_test_new_biggerIW_0-400.nc"
topo1 = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/Tb_pertub_cut_0-400.nc"

v_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/v_TIV_IW35_interval1_maxdisplacement25_400-1600.nc"
u_file = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/u_TIV_IW35_interval1_maxdisplacement25_400-1600.nc"
topo2 = "/home/benjamin/Met_ParametersTST/T0/data/test_tiv/Tb_pertub_cut_400-1600.nc"













file_u1 = h5py.File(u_new_file,'r')
file_topo = h5py.File(topo1,'r')
topo1 = file_topo.get("Tb_pertub")
v = file_v1.get("v")
v = np.array(v)
u = file_u1.get("u")
u = np.array(u)


file_v2 = h5py.File(v_file,'r')
file_u2 = h5py.File(u_file,'r')
file_topo = h5py.File(topo2,'r')
topo2 = file_topo.get("Tb_pertub")
v_2 = file_v2.get("v")
v_2 = np.array(v_2)
u_2 = file_u2.get("u")
u_2 = np.array(u_2)

u.shape
u_2.shape

u = np.append(u,u_2, axis=0)

v = np.append(v,v_2, axis=0)




u1 = u[:,100:120,120:135]
v1 = v[:,100:120,120:135]


def calcwinddirection(v,u): 
    erg_dir_rad = np.arctan2(v, u)
    erg_dir_deg = np.degrees(erg_dir_rad)
    erg_dir_deg_pos = np.where(erg_dir_deg < 0.0, erg_dir_deg+360, erg_dir_deg)
    return(np.round(erg_dir_deg_pos,2))


def calcwindspeed(v,u): 
    ws = np.sqrt((u*u)+(v*v))
    return(np.round(ws,2))



np.mean(u[:,100,125])
np.mean(v[:,100,125]*-1)
wd_T0 = calcwinddirection(v1*-1,u1)

wd_T0_1px = calcwinddirection(v[:,100,125]*-1,u[:,100,125])


breaks = [0 ,20,40,60,  80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,340,360]
plt.hist(np.round(wd_T0.reshape(wd_T0.shape[0]*wd_T0.shape[1]*wd_T0.shape[2]),0), bins=breaks)
plt.hist(np.round(wd_T0_1px,0), bins=breaks)
np.savetxt(datapath+"area_wd_TIV0-1600.csv", np.round(wd_T0.reshape(wd_T0.shape[0]*wd_T0.shape[1]*wd_T0.shape[2]),0), delimiter=",")


# calc windspeed
ws_T0_area = calcwindspeed(v1*-1,u1)
breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4]
ws_T0_area = ws_T0_area*0.175
plt.hist(ws_T0_area.reshape(ws_T0_area.shape[0]*ws_T0_area.shape[1]*ws_T0_area.shape[2]), bins=breaks)



# show area
topo[1][100:120,120:135] = 2
plt.imshow(v1[1])

#scatterplot mean pertub vs. mean windspeed
wd_T0 = calcwinddirection(v*-1*0.175,u*0.175)
ws_T0 = calcwindspeed(v*-1*0.175,u*0.175)
mean_pertub = np.mean(topo,(1,2))
mean_direction = np.mean(wd_T0,(1,2))
mean_speed = np.mean(ws_T0,(1,2))
plt.scatter(mean_pertub,mean_direction)

