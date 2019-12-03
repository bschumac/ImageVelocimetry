#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:41:27 2019

@author: benjamin
"""

import os


path = "/media/benjamin/Seagate Expansion Drive/Himawari/Tier1/highres/"

fls = os.listdir(path)
fls = sorted(fls, key = lambda x: x.rsplit('.', 1)[0])

fls_nename = []
for name in fls:
    src = path+name
    name = list(name)
    name[-15] = "_"
    newname ="".join(name)
    dst = path+newname
    os.rename(src,dst)
    
    