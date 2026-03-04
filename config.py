#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:20:50 2022

@author: ivan
"""

# values of noise std for each parameter in the state matrix
std_noise_8D=[0.25, 0.25, 0.01, 0.01, 1e-2, 1e-2, 1e-3, 1e-3]
std_noise_4D = [0.25, 0.25, 0.01, 0.01]
K=32                            # K is the number of bins for each dimension in the HS histogram
alpha = 20.0                    #exponent to increase the sharpness of the particle weight distribution
prediction = 'weighted_avg'     # Method to compute the final prediction of the object state
motion_model = 'random_noise'   # for [x,y,w,h] Or 'constant' velocity for a 8D vel model [x,y,w,h,vx,vy,v,vh]
noise_beta = 3.0                # scale for Neff metric for noise adaptative funcion