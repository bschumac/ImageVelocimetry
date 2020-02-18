#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:09:55 2020

@author: benjamin
"""



"""
Optris Workflow:
- Acquisition
- Transformation with IRT Analyzer to CSV
- CSV to arr -> remove steady images - to netcdf
- arr to rgb images
    2 Options:
        -> redvalue: Needs adjustment to maximally 255 values. Multiply all values with 10. Neglict negative values by 
                    setting all below 10 to 10, then subtract 10 to reach the maximum of 255 values. 
                    - can be used with all blender stabilization interpolation methods (Bicubic, Biliniar, Nearest neighbor) 
        -> virdris: uses random forest to predict the temperature - can only be used with nearest neighbor interpolation

- Blender stabilization (depending on which method has been used previously)
- Retrieval of stable Temperature values
- Calculation of Interval 
- Calculation of pertubation
- Run TIV
        

Telops Workflow:
- Acquisition
- Transformation with Telops Matlab library to Mat files
- Transformation from mat to NPY files 
- load NPY files to dask array
- calculate std on dask array to find movement in images
    -> if std is too high: Stabilize
- calculate interval
- calculate perturbations
- Run TIV



"""
