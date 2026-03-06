#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:20:50 2022

@author: ivan
"""

# values of noise std for each parameter in the state matrix
std_noise=[0.25, 0.25, 0.01, 0.01, 1e-2, 1e-2, 1e-3, 1e-3]
K=32                            # K is the number of bins for each dimension in the HS histogram
alpha = 20.0                    #exponent to increase the sharpness of the particle weight distribution
prediction = 'weighted_avg'     # Method to compute the final prediction of the object state
Neff_th = 0.3                   # Value of Neff to apply dynamic noise 
noise_beta = 2.0                # scale for Neff metric for noise adaptative funcion