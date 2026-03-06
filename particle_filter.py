#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:06:37 2022

@author: ivan
"""

import numpy as np
from skimage.color import rgb2hsv
import pdb
import numpy.random as npr
import config as cfg

def computeMultiChannelHistogram(im, K):
    """
    Computes a normalized joint histogram for an image with a variable number of channels.
    Assumes that the input values of 'im' are normalized in the range [0.0, 1.0].
    
    Parameters:
    - im: 3-dimensional NumPy array (Height, Width, Channels).
    - K: Number of bins per channel.
    
    Returns:
    - hist: 1D vector containing the normalized joint histogram of size K^Channels.
    """
    # In case we have a 2D image with just one channel 
    if im.ndim == 2:
        im = im[:, :, np.newaxis]
        
    h, w, c = im.shape
    
    # 1. Vectorize the image to have a list of pixels (H*W, C)
    im_flat = np.reshape(im, (h * w, c))
    
    # 2. Quantize the values. 
    # Subtract 1e-30 to ensure the maximum value (1.0) falls into bin K-1 and doesn't go out of bounds.
    r = np.floor((im_flat - 1e-30) * K).astype(int)
    
    # For safety, ensure no index goes out of the [0, K-1] boundaries
    r = np.clip(r, 0, K - 1)
    
    # 3. Compute the linear index for np.bincount (base K to base 10 conversion)
    rlin = np.zeros(h * w, dtype=int)
    for i in range(c):
        # The weight of each channel decreases from left to right (K^(c-1-i))
        rlin += r[:, i] * (K ** (c - 1 - i))
        
    # 4. Compute the frequency histogram
    # The total size of the histogram will be K raised to the number of channels
    total_bins = K ** c
    hist = np.bincount(rlin, minlength=total_bins)
    
    # 5. Normalize to obtain a probability distribution (summing to 1)
    hist = hist / (hist.sum() + 1e-10)
    
    return hist

class particle_filter:

    """
    Class that implements a particle filter with two methods:
        - Constructor: __init__
        - update()
    """
    def __init__(self, im0, bbox,numParticles=100,step=1):
        
        self.K = cfg.K #K is the number of bins in the histogram
        self.N = numParticles #Number of particles
        self.t = step #Delay between frames for the state-transition matrix
        self.alpha = cfg.alpha #exponent to increase the sharpness of the particle weight distribution
        
        
        #Set the initial state X_init=[xstatic,_xdynamic]
        xstatic=np.array(
            [bbox[0]+0.5*bbox[2],       # bb_center_x
             bbox[1]+0.5*bbox[3],       # bb_center_y
             bbox[2],                   # bb_width
             bbox[3],                   # bb_height
             ])
        xdynamic=np.zeros((4,))         # velocity_x, velocity_y, velocity_width, velocity_height

        # State-transition matrix
        self.x_init = np.concatenate((xstatic, xdynamic),axis=0)    # 8D
        self.A=np.block(
                [[np.eye(4), self.t*np.eye(4)],
                 [np.zeros((4,4)), np.eye(4)]]);   
                    
        #We obtain the visual representation of the original object
        self.bbox = np.round(bbox)
        objim = im0[int(self.bbox[1]):int(self.bbox[1]+self.bbox[3]), int(self.bbox[0]):int(self.bbox[0]+self.bbox[2]), :]

        # Compute the reference histogram => work in hsv space
        objim_hsv = rgb2hsv(objim)[:, :, :2]

        #Compute the reference histogram => work in hsv space
        self.hist_ref =  computeMultiChannelHistogram(objim_hsv, self.K)  
        
        #Copy the state to all particles x is NxP being N the number of particles and P the number of parameters 
        self.x=np.tile(self.x_init,(self.N,1));

        #Initialize weights uniformly
        self.w=(1/self.N)*np.ones((self.N,));
        #Cumulative weights for particle resampling
        self.c=np.cumsum(self.w);

        #Vector with standard deviations of additive gaussian noise
        #Each dimension corresponds with one element in the state
        self.Sigma = np.array(cfg.std_noise)
        #Make sigma of static variables proportional to bounding box size
        self.Sigma[:2]=self.Sigma[:2]*np.min(self.bbox[2:4]);
        self.Sigma[2:4]=self.Sigma[2:4]*self.bbox[2:4];


    def update(self, im):
        
        # Number of params in the state
        P = self.x.shape[1] 
        
        # Dimensions of the frame
        height, width, colors = im.shape
        
        #####STEP 1: PARTICLE RESAMPLING#####
        # Generate N random values between 0 and 1
        vals = npr.rand(self.N)
        #Choose particle indexes with a value large than vals
        idx_particles = np.searchsorted(self.c, vals)

        # u0 = npr.rand() / self.N
        # u = u0 + np.arange(self.N) / self.N
        # idx_particles = np.searchsorted(self.c, u)
        # Get particle states
        x_past = self.x[idx_particles, :]
        
        #####STEP 2: UPDATE THE PARTICLE STATE#######
        x_new = self.particle_update_(x_past, P)
        
        ##### STEP 3: EXTRACT THE CANDIDATE AREA ########
        im_hsv = rgb2hsv(im)[:, :, :2]
        
        # Particles loop (this cannot be paralelized)
        for i in range(self.N):
            ##### STEP 4: EXTRACT THE CANDIDATE AREA ########
            try:
                # We extract the region in the image corresponding with the bounding box
                limy = np.array([np.ceil(x_new[i,1]-0.5*x_new[i,3]), np.floor(x_new[i,1]+0.5*x_new[i,3])], dtype=int)
                limy = np.clip(limy, 0, height)
                limx = np.array([np.ceil(x_new[i,0]-0.5*x_new[i,2]), np.floor(x_new[i,0]+0.5*x_new[i,2])], dtype=int)
                limx = np.clip(limx, 0, width)
                candidate_reg = im_hsv[limy[0]:limy[1], limx[0]:limx[1], :] 
                
                ##### STEP 5: COMPUTE THE COLOR HISTOGRAM ###########
                
                # If size=0
                if candidate_reg.size == 0:
                    hist = np.zeros_like(self.hist_ref)
                else:
                    hist = computeMultiChannelHistogram(candidate_reg, self.K)
            except:
                hist = np.zeros_like(self.hist_ref)
        
            ########### STEP 6: COMPUTE THE BATTACHARYYA COEFICCIENT ###########
            hist_intersect = (np.sqrt(self.hist_ref * hist)).sum()
            
            # Update the weight of the particle
            self.w[i] = hist_intersect ** self.alpha   
        
        ########### STEP 7: UPDATE X, NORMALIZE THE WEIGHTS AND RECOMPUTE C ###########
        self.x = x_new
        if self.w.sum() > 1e-30:
           self.w = self.w / self.w.sum()
        else:
            print('Error')
            self.w[...] = 1 / self.N
        
        self.c = np.cumsum(self.w)
        
        ########### STEP 8: ESTIMATE THE BOUNDING BOX FROM THE PARTICLES ###########
        # Weighted average
        if cfg.prediction == 'weighted_avg':
            x_global = np.sum(self.w[:, np.newaxis] * self.x, axis=0)
        # Best particle
        elif cfg.prediction == 'max':
            idx_particle = np.argmax(self.w)
            x_global = self.x[idx_particle, ...]
        
        self.bbox = np.array([x_global[0]-0.5*x_global[2], x_global[1]-0.5*x_global[3], x_global[2], x_global[3]])

    def particle_update_(self, x_past, P):

        # Dinamic noise with Neff
        Neff = 1.0 / np.sum(self.w**2)
        Neff /= self.N

        # Neff_threshold = 0.5
        # print(
        #     f"Neff {Neff:.3f} | "
        #     f"maxW {np.max(self.w):.3f} | "
        #     f"stdW {np.std(self.w):.3f} | "
        #     f"vel ({x_past[:,4].mean():.2f},{x_past[:,5].mean():.2f})",
        #     end="\t"
        # )

        dinamic_noise = self.Sigma.copy()
        if Neff < cfg.Neff_th:
            scale = 1 + cfg.noise_beta * (1 - Neff)
            dinamic_noise *= scale

        noise_all = npr.randn(self.N, P) * dinamic_noise

        # Compute the new state updating the previous one
        x_new = (self.A @ x_past.T).T + noise_all

        return x_new
