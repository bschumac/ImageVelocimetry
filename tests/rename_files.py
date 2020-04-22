#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:44:26 2020

@author: benjamin
"""


import os

fld = "/mnt/Seagate_Drive1/Tasman_OP/smooth_converted/"

fld_lst = os.listdir(fld)



for filename in fld_lst:
    #filename = fld_lst[0]
    path, name  = os.path.split(fld+filename)
    name, extension = os.path.splitext(name)
    newname = name+"_smooth"+extension
    os.rename(fld+filename, fld+newname)