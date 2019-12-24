#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:03:18 2019

@author: benjamin
"""

import numpy as np
from scipy import signal
import h5py
import matplotlib.pyplot as plt
from functions.TST_fun import *


def lucas_kanade_np(im1, im2, win=2):
    assert im1.shape == im2.shape
    I_x = np.zeros(im1.shape)
    I_y = np.zeros(im1.shape)
    I_t = np.zeros(im1.shape)
    I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2
    I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2
    I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
    params = np.zeros(im1.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
    params[..., 0] = I_x * I_x # I_x2
    params[..., 1] = I_y * I_y # I_y2
    params[..., 2] = I_x * I_y # I_xy
    params[..., 3] = I_x * I_t # I_xt
    params[..., 4] = I_y * I_t # I_yt
   
    del I_x, I_y, I_t
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    del params
    
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])
    del cum_params
    op_flow = np.zeros(im1.shape + (2,))
    det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
    op_flow_x = np.where(det != 0,
                         (win_params[..., 1] * win_params[..., 3] -
                          win_params[..., 2] * win_params[..., 4]) / det,
                         0)

    op_flow_y = np.where(det != 0,
                         (win_params[..., 0] * win_params[..., 4] -
                          win_params[..., 2] * win_params[..., 3]) / det,
                         0)
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
    op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
    return op_flow


import numpy as np
from matplotlib import pyplot as plt
import cv2

def reduce(image, level = 1):
    result = np.copy(image)
    for _ in range(level-1):
        result = cv2.pyrDown(result)
    return result

def expand(image, level = 1):
    return cv2.pyrUp(np.copy(image))

def compute_flow_map(u, v, gran = 8):
    flow_map = np.zeros(u.shape)

    for y in range(flow_map.shape[0]):
        for x in range(flow_map.shape[1]):
            
            if y%gran == 0 and x%gran == 0:
                dx = 10*int(u[y,x])
                dy = 10*int(v[y,x])

                if dx > 0 or dy > 0:
                    cv2.arrowedLine(flow_map, (x,y), (x+dx,y+dy), 255, 1)
                    
    return flow_map







def lucas_kanade_np(im1, im2, win=7):
    I_x = np.zeros(im1.shape)
    I_y = np.zeros(im1.shape)
    I_t = np.zeros(im1.shape)
    
    I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2])/2
    I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1])/2
    I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
    
    params = np.zeros(im1.shape + (5,))
    params[..., 0] = cv2.GaussianBlur(I_x * I_x, (5,5), 3)
    params[..., 1] = cv2.GaussianBlur(I_y * I_y, (5,5), 3)
    params[..., 2] = cv2.GaussianBlur(I_x * I_y, (5,5), 3)
    params[..., 3] = cv2.GaussianBlur(I_x * I_t, (5,5), 3)
    params[..., 4] = cv2.GaussianBlur(I_y * I_t, (5,5), 3)
    
    cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
    win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                  cum_params[2 * win + 1:, :-1 - 2 * win] -
                  cum_params[:-1 - 2 * win, 2 * win + 1:] +
                  cum_params[:-1 - 2 * win, :-1 - 2 * win])

    u = np.zeros(im1.shape)
    v = np.zeros(im1.shape)
    
    I_xx =  win_params[..., 0]
    I_yy =  win_params[..., 1]
    I_xy =  win_params[..., 2]
    I_xt = -win_params[..., 3]
    I_yt = -win_params[..., 4]        
    
    M_det = I_xx*I_yy - I_xy**2
    temp_u = I_yy*   (-I_xt) + (-I_xy)*(-I_yt)
    temp_v = (-I_xy)*(-I_xt) +    I_xx*(-I_yt)
    op_flow_x = np.where(M_det != 0, temp_u/M_det, 0)
    op_flow_y = np.where(M_det != 0, temp_v/M_det, 0)
    
    u[win + 1: -1 - win, win + 1: -1 - win] = op_flow_x[:-1, :-1]
    v[win + 1: -1 - win, win + 1: -1 - win] = op_flow_y[:-1, :-1]
    
    return u, v















