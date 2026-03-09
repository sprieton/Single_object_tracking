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
dinamic_Neff_th = 0.3                   # Value of Neff to apply dynamic noise 
noise_beta = 2.0                # scale for Neff metric for noise adaptative funcion
mcmc_expl_fact = 1.0            # exploration factor to the MCMC particle exploration
update_new_inf = 0.02           # Update a 2% of the old reference with the new mask
hist_update_th = 0.9            # how good must be the new histogram to be updated reference
lost_obj_Neff_th = 0.00         # Neff value under threshold consider we lost object
lost_obj_part_restart = 0.1     # Number of particles randomized when lost object

# flags
DEBUG = False