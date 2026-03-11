#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 15:20:50 2022

@author: ivan
"""

# =============================================================================
# 1. General & State Initialization
# =============================================================================
std_noise=[0.25, 0.25, 0.01, 0.01, 1e-2, 1e-2, 1e-3, 1e-3] # Original, more precise noise
K=32                            # K is the number of bins for each dimension in the HS histogram
prediction = 'weighted_avg'     # Options: 'max', 'weighted_avg', 'robust_mean'
pred_min_frac = 0.3             # Min fraction of particles used in weighted_avg (0.0 to 1.0)

# =============================================================================
# 2. Observation Model
# =============================================================================
alpha = 20.0                    # Original, balanced sharpness
observation_model = 'ellipse_hist' # Options: 'spatial_hist' (default), 'ellipse_hist', 'deep_learning'
ellipse_area_ratio = 0.5        # Fraction of the bbox area for the ellipse (0.5 = 50% inner, 50% outer)
dl_input_size = (64, 64)        # Input size for the neural network (smaller = faster)
dl_weight = 0.5                 # Weight of Deep Learning vs Color Histogram (0.5 = 50% each)
device = 'cuda'                 # 'cpu' or 'cuda'

# =============================================================================
# 3. Adaptive Noise & MCMC
# =============================================================================
dinamic_Neff_th = 0.3           # Value of Neff to apply dynamic noise 
noise_beta = 2.0                # scale for Neff metric for noise adaptative funcion
update_new_inf = 0.02           # Update a 2% of the old reference with the new mask
speed_noise_factor = 0.5
num_frames_vel=6                # number of frames to start to read velocity
mcmc_expl_fact = 1.0            # exploration factor to the MCMC particle exploration
speed_mcmc_factor = 0.5         # speed factor to the MCMC particle exploration
t_df_max = 20       # df máximo
t_df_min = 2        # df mínimo
t_df_speed_factor = 5.0

# =============================================================================
# 4. Reference Update (Appearance Model)
# =============================================================================
update_new_inf = 0.02           # Base update rate (beta) for the reference histogram
hist_update_th = 1.1            # Set > 1.0 to disable histogram updates (Static Model)
anchor_weight = 0.0             # Not used

# =============================================================================
# 5. Recovery / Lost Object Strategy
# =============================================================================
lost_obj_Neff_th = 0.00         # Disable recovery to maximize precision (Best Results state)
lost_obj_part_restart = 0.1     # Percentage of particles to restart/randomize when object is lost

# =============================================================================
# 6. Debugging
# =============================================================================
DEBUG = False