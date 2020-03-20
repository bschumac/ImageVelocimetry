#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:33:29 2020

@author: benjamin
"""

from functions.TST_fun import readnetcdftoarr

Tb_stab = readnetcdftoarr("/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/Tb_stab_virdiris_20Hz.nc")

Tb_org = readnetcdftoarr("/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Tb_org.nc")


import ctypes as ct 

libir = ct.cdll.LoadLibrary(ct.util.find_library("irdirectsdk"))

image_ini = libir.evo_irimager_usb_init("/home/benjamin/test2.xml",0,0)


libir.getTemprangeDecimal()

#libir.evo_irimager_get_thermal_image_size()