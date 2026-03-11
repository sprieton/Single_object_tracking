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
        self.hist_ref = computeMultiChannelHistogram(objim_hsv, self.K)  
        
        #Copy the state to all particles x is NxP being N the number of particles and P the number of parameters 
        self.x=np.tile(self.x_init,(self.N,1));

        #Initialize weights uniformly
        self.w=(1/self.N)*np.ones((self.N,));
        #Cumulative weights for particle resampling
        self.c=np.cumsum(self.w);
        self.v_global = np.zeros(2)

        #Vector with standard deviations of additive gaussian noise
        #Each dimension corresponds with one element in the state
        self.Sigma = np.array(cfg.std_noise)
        #Make sigma of static variables proportional to bounding box size
        self.Sigma[:2]=self.Sigma[:2]*np.min(self.bbox[2:4]);
        self.Sigma[2:4]=self.Sigma[2:4]*self.bbox[2:4];


    def update(self, im):
        
        # Number of params in the state
        P = self.x.shape[1] 
        
        #####STEP 1: PARTICLE RESAMPLING#####
        # Generate N random values between 0 and 1
        vals = npr.rand(self.N)
        #Choose particle indexes with a value large than vals
        idx_particles = np.searchsorted(self.c, vals)
        # Get particle states
        x_past = self.x[idx_particles, :]
        self.vel_history = []
        
        #####STEP 2: UPDATE THE PARTICLE STATE#######
        x_new = self.particle_update_(x_past, P, im.shape)
        
        ##### STEP 3: UPDATE THE WEIGHTS ########
        self.v_global = np.average(self.x[:,4:6], axis=0, weights=self.w)   # update the velocity of the system
        im_hsv = rgb2hsv(im)[:, :, :2]
        self.weight_update_mcmc_(x_new, x_past, P, im_hsv)
        
        ########### STEP 4: NORMALIZE WEIGHTS AND RECOMPUTE C ###########
        self.x = x_new
        self.w = np.clip(self.w, 1e-10, None)
        self.w /= self.w.sum()
        self.c = np.cumsum(self.w)
        
        ########### STEP 5: ESTIMATE THE BOUNDING BOX FROM THE PARTICLES ###########
        Neff = 1.0 / np.sum(self.w**2) / self.N

        if cfg.prediction == 'weighted_avg':
            frac_particles = np.clip(Neff / cfg.dinamic_Neff_th, 0.3, 1.0)
            if cfg.DEBUG:
                print(f"avg of {frac_particles:.3f}%", end=" | ")
            n_use = max(1, int(self.N * frac_particles))
            idx_top = np.argsort(self.w)[-n_use:]
            w_top = self.w[idx_top]
            w_top /= w_top.sum()
            x_global = np.sum(w_top[:, np.newaxis] * self.x[idx_top, :], axis=0)

        elif cfg.prediction == 'max':
            idx_particle = np.argmax(self.w)
            x_global = self.x[idx_particle, ...]

        # Update bbox from x_global
        self.bbox = np.array([
            x_global[0] - 0.5 * x_global[2],
            x_global[1] - 0.5 * x_global[3],
            x_global[2],
            x_global[3]
        ])

    def particle_update_(self, x_past, P, im_shape):
        """
        Updates the particles using a Resample-Move strategy with MCMC.

        Parameters
        ----------
        x_past : ndarray
            Resampled particles (NxP).
        P : int
            Number of state dimensions.
        """
        # 1- Dinamic noise with Neff
        Neff = 1.0 / np.sum(self.w**2)
        Neff /= self.N

        # apply dinamic noise to the deprecated particles if necessary
        dinamic_noise = self.Sigma.copy()
        if Neff < cfg.dinamic_Neff_th:
            scale = 1 + cfg.noise_beta * (1 - Neff)
            dinamic_noise *= scale

        if cfg.DEBUG:
            print(
                f"Neff {Neff:.3f} | "
                f"maxW {np.max(self.w):.3f} | "
                f"stdW {np.std(self.w):.3f} | "
                f"vel ({x_past[:,4].mean():.2f},{x_past[:,5].mean():.2f})",
                end="\t"
            )
        
        # 2- Apply linear motion model
        x_new = (self.A @ x_past.T).T

        # 3- Compute average velocity and magnitude
        vel_mean = np.mean(x_past[:, 4:6], axis=0)            # 2D velocity vector
        vel_mag = np.linalg.norm(vel_mean)                   # magnitude
        vel_dir = vel_mean / (vel_mag + 1e-10)              # normalized direction
        self.vel_history.append(vel_mag)
        if len(self.vel_history) > cfg.num_frames_vel:
            self.vel_history.pop(0)

        # 4- Determine speed factor and adaptive t-Student degrees of freedom
        if len(self.vel_history) < cfg.num_frames_vel:
            speed_factor = 1.0
            df = cfg.t_df_max
        else:
            avg_speed = np.mean(self.vel_history)
            speed_factor = 1.0 + cfg.speed_noise_factor * avg_speed
            # Adaptive df: higher df → lower speed, lower df → higher speed
            df = np.clip(cfg.t_df_max / (1.0 + cfg.t_df_speed_factor * avg_speed), 
                         cfg.t_df_min, cfg.t_df_max)

        # 5- Generate base t-Student noise for position (2D: x,y)
        noise_base = npr.standard_t(df, size=(self.N, 2))

        # 6- Compute perpendicular direction for lateral exploration
        perp_dir = np.array([-vel_dir[1], vel_dir[0]])

        # 7- Scale noise along parallel and perpendicular directions
        scale_parallel = dinamic_noise[:2] * speed_factor
        scale_perp = dinamic_noise[:2] * speed_factor * 0.3  # smaller lateral component

        # 8- Deform noise along motion direction
        noise_pos = (noise_base[:, 0][:, None] * vel_dir * scale_parallel) + \
                    (noise_base[:, 1][:, None] * perp_dir * scale_perp)

        # 9- Apply noise to bounding box width/height and velocity (unmodified t-Student)
        noise_size = npr.standard_t(df, size=(self.N, 2)) * dinamic_noise[2:4] * speed_factor
        noise_vel = npr.standard_t(df, size=(self.N, 4)) * dinamic_noise[4:] * speed_factor

        # 10- Update particle states
        x_new[:, :2] += noise_pos       # x,y positions
        x_new[:, 2:4] += noise_size     # width, height
        x_new[:, 4:] += noise_vel       # velocities

        return x_new
        
    def weight_update_mcmc_(self, x_new, x_past, P, im_hsv):
        """
        Applies a Metropolis-Hastings MCMC move to refine particle states after
        propagation. Proposals are generated via Gaussian perturbations and
        accepted based on the Bhattacharyya likelihood ratio, improving the
        approximation of the posterior p(x_t | z_t).
        """
        n_accepted = 0
        p_proposal_mean = 0
        p_curr_mean = 0
        max_w = np.max(self.w) + 1e-10

        # Propagate the particles with MCMC 
        # Particles loop (this cannot be paralelized)
        for i in range(self.N):
            # 1- Proposal of particle movment inverse proportional to the weight g=Gaussian
            # + weight - MCMC step | - weight + MCMC step
            w_norm = self.w[i] / max_w     # Normalized version
            
            # proportions of the exploration
            speed = np.linalg.norm(x_new[i][4:6])
            scale = cfg.mcmc_expl_fact * (1 - w_norm) * (1 + cfg.speed_mcmc_factor * speed)
           
            # 2- Create the proposal using the scales of ecploration
            proposal = x_new[i].copy()
            noise = npr.randn(P) * scale * self.Sigma

            # velocity guided exploration
            vel = x_new[i][4:6]
            vel_norm = vel / (np.linalg.norm(vel) + 1e-10)
            noise[:2] += vel_norm * scale

            proposal += noise
            
            # 3- Limit the proposal inisde the image
            proposal[0] = np.clip(proposal[0], 0, im_hsv.shape[1])
            proposal[1] = np.clip(proposal[1], 0, im_hsv.shape[0])
            proposal[2:4] = np.clip(proposal[2:4], 1, np.min(np.array(im_hsv.shape[0:2])))
            
            # 4- Get the candidate likelihoods
            hist_prop = self.get_particle_hist_(proposal, im_hsv)
            hist_curr = self.get_particle_hist_(x_new[i], im_hsv)
            
            p_proposal = self.get_Battacharyya_(hist_prop) ** self.alpha
            p_current = self.get_Battacharyya_(hist_curr) ** self.alpha
            p_proposal_mean += p_proposal
            p_curr_mean += p_current

            # 5- add Dynamic likelihood of proposal and original velocity
            # vel_prop = proposal[4:6]
            # vel_curr = x_new[i][4:6]

            # motion_prop = np.exp(-0.5*np.sum((vel_prop-self.v_global)**2)/cfg.motion_sigma)
            # motion_curr = np.exp(-0.5*np.sum((vel_curr-self.v_global)**2)/cfg.motion_sigma)

            # p_proposal *= motion_prop
            # p_current *= motion_curr

            # 6- Use Metropolis Hastings to update the particles proporsals
            # Accpetance ratio α = p(x′)/p(xt​) ​(Metropolis-Hastings)
            accept_ratio = min(1.0, p_proposal / (p_current + 1e-10))

            # Accept probability u ≤ α / u∼U(0,1) and update the weights
            log_ratio = np.log(p_proposal + 1e-12) - np.log(p_current + 1e-12)
            accept_ratio = min(1.0, np.exp(log_ratio))

            if npr.rand() < accept_ratio:
                x_new[i] = proposal # Accept the proposal
                self.w[i] = p_proposal  
                n_accepted += 1

            else:   # we keep the original particle
                self.w[i] = p_current
            # if npr.rand() <= accept_ratio:
            #     x_new[i] = proposal  
            #     self.w[i] = p_proposal
            #     n_accepted += 1
            # else: 
            #     self.w[i] = p_current

        p_proposal_mean /= self.N
        p_curr_mean /= self.N
        
        if cfg.DEBUG:
            print(f"Accpeted: ({n_accepted}/{self.N})", end="")
    
    def get_particle_hist_(self, particle, im_hsv):
        """
        Extracts the image region defined by a particle bounding box and
        computes its HSV color histogram.

        Parameters
        ----------
        particle : array-like
            State vector of the particle [x, y, width, height, ...].
        im_hsv : ndarray
            Input image in HSV space.

        Returns
        -------
        hist : ndarray
            Normalized histogram of the candidate region.
        """

        height, width, _ = im_hsv.shape

        # Particle parameters
        x, y, w, h = particle[:4]

        try:
            # Compute bounding box limits
            limy = np.array([np.ceil(y - 0.5 * h), np.floor(y + 0.5 * h)], dtype=int)
            limy = np.clip(limy, 0, height)

            limx = np.array([np.ceil(x - 0.5 * w), np.floor(x + 0.5 * w)], dtype=int)
            limx = np.clip(limx, 0, width)

            # Extract candidate region
            candidate_reg = im_hsv[limy[0]:limy[1], limx[0]:limx[1], :]

            # Compute histogram
            if candidate_reg.size == 0:
                hist = np.zeros_like(self.hist_ref)
            else:
                hist = computeMultiChannelHistogram(candidate_reg, self.K)

        except Exception:
            hist = np.zeros_like(self.hist_ref)

        return hist

    def get_Battacharyya_(self, hist):
        return (np.sqrt(self.hist_ref * hist)).sum()