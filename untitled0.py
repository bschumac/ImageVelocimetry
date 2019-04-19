#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:20:48 2019

@author: benjamin
"""
image_type1 = ".tif"
image_type2 = ".tif"
org_datapath = "/home/benjamin/Met_ParametersTST/T1/"
img_datapath1 = "/home/benjamin/Met_ParametersTST/T1/Tier01/12012019/Optris_data/Flight03_O80_1616_tif/"
img_datapath2="/home/benjamin/Met_ParametersTST/T1/Tier02/12012019/Optris_data/stab_turf5/"
n = 0
print(n)
org_rbg_path = img_datapath1
image_file=org_rbg_path+str(n)+image_type1
org_data_rgb=plt.imread(image_file)
org_data_rgb=org_data_rgb[:,:,0:3]

stab_rbg_path = img_datapath2
image_file=stab_rbg_path+format(n, '04d')+image_type2
stab_data_rgb=plt.imread(image_file)
stab_data_rgb=stab_data_rgb[:,:,0:3]


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = org_data_rgb
img2 = img.copy()
template = stab_data_rgb
w, h = template.shape[1], template.shape[0]

# All the 6 methods for comparison in a list
#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

#for meth in methods:
img = img2.copy()
meth = 'cv2.TM_SQDIFF'
method = eval(meth)

# Apply template Matching
res = cv2.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc

bottom_right = (top_left[0] + w, top_left[1] + h)

bottom_left = (top_left[0] + w, top_left[1] + h) 

top_right = (top_left[0],top_left[1] + h)




cv2.rectangle(img,top_left, bottom_right, 255, 2)
plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(meth)

plt.show()