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
update_new_inf = 0.02           # Update a 2% of the old reference with the new mask
speed_noise_factor = 0.5
num_frames_vel=6                # number of frames to start to read velocity
mcmc_expl_fact = 1.0            # exploration factor to the MCMC particle exploration
speed_mcmc_factor = 0.5         # speed factor to the MCMC particle exploration
t_df_max = 20       # df máximo
t_df_min = 2        # df mínimo
t_df_speed_factor = 5.0

# flags
DEBUG = False