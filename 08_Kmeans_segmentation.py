#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:04:38 2018

@author: benjamin
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import h5py
from matplotlib import colors, cm
from TST_fun import tst_kmeans

n_clusters = 8
my_dpi = 100


datapath = "/home/benjamin/Met_ParametersTST/T0/data/"


file = h5py.File(datapath+'Tb_stab_rect_pertub_py.nc','r')
tb= file.get("Tb_pertub")

labels= []
centers = []

tst_kmeans(tb,outpath=datapath)




