#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 19:52:21 2018

@author: benjamin
"""

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*6,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:4].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Find the chess board corners
#ret, corners = cv.findChessboardCorners(img, (6,4), None)
## If found, add object points, image points (after refining them)
#if ret == True:
#    objpoints.append(objp)
#    corners2 = cv.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
#    imgpoints.append(corners)
#    # Draw and display the corners
#    cv.drawChessboardCorners(img, (6,4), corners2, ret)
#    cv.imshow('img', img)
#    cv.waitKey(500)
#cv.destroyAllWindows()
#
#cv.drawChessboardCorners(img, patternsize, Mat(corners), patternfound)
#plt.imshow(img)




#
## mouse callback function
# =============================================================================
# def draw_circle(event,x,y,flags,param):
#     global ix,iy
#     global pointIndex
#     global pts
# 
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         #cv2.circle(img,(x,y),100,(255,0,0),-1)
#         ix,iy = x,y
#         pts[pointIndex] = (x,y)
#         pointIndex = pointIndex + 1
# 
# =============================================================================
# Create a black image, a window and bind the function to window
#cv2.namedWindow('image')
#cv2.setMouseCallback('image',draw_circle)
#while(pointIndex != 4):
#    cv2.imshow('image',img)
#    k = cv2.waitKey(20) & 0xFF
#    if k == 27:
#        break





import cv2
import numpy as np
import glob
images = glob.glob('/home/benjamin/Met_ParametersTST/T0/data/figure_Tb_RBG_stab/*.png')
import matplotlib.pyplot as plt
import time
import progressbar as pb
import random
from matplotlib import colors, cm
import os

filelst = os.listdir("/home/benjamin/Met_ParametersTST/T0/data/figure_Tb_RGB_tif_rect_stab/")
bar4 = pb.ProgressBar(maxval=len(filelst), widgets=[pb.Bar('=', '[', ']'), ' ', pb.Percentage()]) 
bar4.start()
bar4_iterator = 0



def getint(name):
    basename = name.partition('.')
    num = basename[0]
    return int(num)
getint(filelst[1])

filelst.sort(key=getint)


for k in range(0,len(filelst)):
#for fname in images:
    
    fname = "/home/benjamin/Met_ParametersTST/T0/data/figure_Tb_RGB_tif_rect_stab/"+filelst[k]
    
    
    pointIndex = 0
    
    ix,iy = -1,-1
    ASPECT_RATIO = (382,288)
    
    pts2 = np.float32([[0,0],[ASPECT_RATIO[1],0],[0,ASPECT_RATIO[0]],[ASPECT_RATIO[1],ASPECT_RATIO[0]]])
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    img2 = np.flip(img,2)
            
    pts = [(0,0),(235,0),(0,450),(630,300)]        
    
    pts1 = np.float32([\
    			[pts[0][0],pts[0][1]],\
    			[pts[1][0],pts[1][1]],\
    			[pts[2][0],pts[2][1]],\
    			[pts[3][0],pts[3][1]] ])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    
    dst = cv2.warpPerspective(img,M,ASPECT_RATIO)
    
    #ax1 = plt.subplot(111)
    #im=ax1.imshow(u_xz_norm,interpolation=None,cmap=cm.jet)
    dst2 = np.flip(dst,2)

    image_file='/home/benjamin/Met_ParametersTST/T0/data/figure_Tb_RGB_tif_rect_stab_rectOCV/'+format(k, '04d')+'.png'  
    
    plt.imsave(image_file,dst2)   
        

    #image_file='/home/benjamin/Met_ParametersTST/T0/data/rectify/'+format(k, '04d')+'.png'  
    
    #plt.imsave(image_file,dst2)    

    bar4.update(bar4_iterator+1)
    bar4_iterator += 1
    
bar4.finish()




# anstatt getPerspectiveTransform koennte auch findHomography(srcPts, dstPts) nuetzlich sein
# die Output matrix anstatt M verwenden:

# muss getestet werden

#M = cv2.getPerspectiveTransform(pts1,pts2)

#dst1 = cv2.warpPerspective(img,M,ASPECT_RATIO)
#
#image_file='/home/benjamin/Met_ParametersTST/T0/data/test_rectify/'+format(x, '04d')+'.png'  
#    
#plt.imsave(image_file,dst1) 





