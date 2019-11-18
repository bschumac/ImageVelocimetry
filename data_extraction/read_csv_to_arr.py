#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 11:14:40 2019

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
from scipy import stats
import datetime
from math import sqrt
#import openpiv.filters
import os
import copy
import scipy
import pathos
import skimage



tb_freezer = readcsvtoarr("/home/benjamin/Met_ParametersTST/Lab_freezer_test/Tier1/csv/", end_img = 10000)





writeNetCDF("/home/benjamin/Met_ParametersTST/Lab_freezer_test/Tier1/", "freezer_10000.nc", "Tb", tb_freezer)

tb_freezer_pertub = create_tst_pertubations_mm(tb_freezer, 135)


writeNetCDF("/home/benjamin/Met_ParametersTST/Lab_freezer_test/Tier1/", "freezer_10000_pertub.nc", "Tb_pertub", tb_freezer_pertub)
