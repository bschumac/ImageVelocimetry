#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:33:29 2020

@author: benjamin
"""

from functions.TST_fun import readnetcdftoarr

Tb_stab = readnetcdftoarr("/home/benjamin/Met_ParametersTST/T1/Tier03/12012019/Optris_data/Flight03_O80_1616/Tb_stab_virdiris_20Hz.nc")

Tb_org = readnetcdftoarr("/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/Tb_org.nc")


from ctypes import *

libir = cdll.LoadLibrary(util.find_library("irdirectsdk"))

libir.evo_irimager_usb_init()