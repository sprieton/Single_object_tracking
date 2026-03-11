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
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    import torch.nn.functional as F
    HAS_TORCH = True
except (ImportError, RuntimeError):
    HAS_TORCH = False
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
        # Spatial Histogram: Split into Top and Bottom
        h_obj, _, _ = objim_hsv.shape
        h_half = h_obj // 2
        hist_top = computeMultiChannelHistogram(objim_hsv[:h_half, :, :], self.K)
        hist_bottom = computeMultiChannelHistogram(objim_hsv[h_half:, :, :], self.K)
        self.hist_ref = np.concatenate((hist_top, hist_bottom)) / 2.0
        self.hist_init = self.hist_ref.copy()
        
        # Deep Learning Initialization
        self.dl_ref = None
        self.dl_model = None
        self.transform = None
        
        if cfg.observation_model == 'deep_learning':
            if not HAS_TORCH:
                raise ImportError("PyTorch is required for deep_learning mode. Install it via pip.")
            
            # Load lightweight model
            self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
            print(f"Deep Learning Model loaded on: {self.device}")
            # Use MobileNetV3 Small - extremely fast
            self.dl_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(self.device)
            self.dl_model.eval() # Set to evaluation mode
            
            # Preprocessing transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(cfg.dl_input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Compute reference embedding
            # objim is RGB (from object_tracking.py)
            self.dl_ref = self.compute_deep_embedding_single_(objim)

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
        
        #####STEP 1: PARTICLE RESAMPLING#####
        # Generate N random values between 0 and 1
        vals = npr.rand(self.N)
        #Choose particle indexes with a value large than vals
        idx_particles = np.searchsorted(self.c, vals)
        # Get particle states
        x_past = self.x[idx_particles, :]
        
        #####STEP 2: UPDATE THE PARTICLE STATE#######
        x_new = self.particle_update_(x_past, P, im.shape)
        
        ##### STEP 3: UPDATE THE WEIGHTS ########
        if cfg.observation_model == 'deep_learning':
            # For Deep Learning, we skip MCMC because running the network iteratively is too slow.
            # We evaluate all particles in a BATCH.
            
            # 1. Compute Color Likelihoods (Fast)
            im_hsv = rgb2hsv(im)[:, :, :2]
            color_weights = self.compute_color_likelihoods_batch_(x_new, im_hsv)
            
            # 2. Compute Deep Learning Likelihoods (Batch Processing)
            dl_weights = self.compute_deep_likelihoods_batch_(x_new, im)
            
            # 3. Combine
            self.w = (1 - cfg.dl_weight) * color_weights + cfg.dl_weight * dl_weights
        else:
            im_hsv = rgb2hsv(im)[:, :, :2]
            self.weight_update_mcmc_(x_new, P, im_hsv)
        
        ########### STEP 4: NORMALIZE WEIGHTS AND RECOMPUTE C ###########
        self.x = x_new
        self.w = np.clip(self.w, 1e-10, None)
        self.w /= self.w.sum()
        self.c = np.cumsum(self.w)
        
        ########### STEP 5: ESTIMATE THE BOUNDING BOX FROM THE PARTICLES ###########
        Neff = 1.0 / np.sum(self.w**2) / self.N

        if cfg.prediction == 'weighted_avg':
            frac_particles = np.clip(Neff / cfg.dinamic_Neff_th, cfg.pred_min_frac, 1.0)
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
            
        elif cfg.prediction == 'robust_mean':
            # 1. Initial weighted estimate using all particles
            x_mean = np.average(self.x, weights=self.w, axis=0)
            
            # 2. Filter spatial outliers (particles too far from the center)
            # Threshold: half of the object diagonal
            dists = np.linalg.norm(self.x[:, :2] - x_mean[:2], axis=1)
            threshold = np.linalg.norm(x_mean[2:4]) * 0.5
            mask = dists < threshold
            
            # 3. Recompute weighted average with inliers only
            w_masked = self.w[mask] + 1e-30 # Avoid division by zero
            x_global = np.average(self.x[mask], weights=w_masked, axis=0)

        
        # Update bbox from x_global
        self.bbox = np.array([
            x_global[0] - 0.5 * x_global[2],
            x_global[1] - 0.5 * x_global[3],
            x_global[2],
            x_global[3]
        ])

        ##### STEP 6: UPDATE REFERENCE HISTOGRAM ADAPTIVE #####
        self.update_hist_ref(im_hsv, x_global, Neff, cfg.update_new_inf)

    def particle_update_(self, x_past, P, im_shape):
        """
        Actualiza las partículas usando Resample-Move con MCMC.
        x_past: partículas remuestreadas (NxP)
        P: número de dimensiones del estado
        """
        # 1- Dinamic noise with Neff
        Neff = 1.0 / np.sum(self.w**2)
        Neff /= self.N

        # apply dinamic noise to the deprecated particles if necessary
        dinamic_noise = self.Sigma.copy()
        if cfg.lost_obj_Neff_th < Neff < cfg.dinamic_Neff_th:
            scale = 1 + cfg.noise_beta * (1 - Neff)
            dinamic_noise *= scale

        ## In case we lost object we restart some particles
        elif Neff < cfg.lost_obj_Neff_th: 
            num_reset = int(cfg.lost_obj_part_restart * self.N)
            H, W = im_shape[:2]
            x_global = x_past.mean(axis=0)  # last known position

            for idx in range(num_reset):
                # Gaussian distribution of the restarted particles to search the object
                x_past[idx, 0] = np.clip(x_global[0] + (npr.randn() * 0.2 * W), 0, W)    # x 
                x_past[idx, 1] = np.clip( x_global[1] + (npr.randn() * 0.2 * H), 0, H)   # y 
                # width and height as the last known position + little noise
                x_past[idx, 2] = np.clip(x_global[2] + npr.randn() * 0.1 * x_global[2], 1, W)  # width
                x_past[idx, 3] = np.clip(x_global[3] + npr.randn() * 0.1 * x_global[3], 1, H)  # height
                # velocities = 0
                x_past[idx, 4:] = 0
            if cfg.DEBUG:
                print(f"Restarted {cfg.lost_obj_part_restart}%")

        if cfg.DEBUG:
            print(
                f"Neff {Neff:.3f} | "
                f"maxW {np.max(self.w):.3f} | "
                f"stdW {np.std(self.w):.3f} | "
                f"vel ({x_past[:,4].mean():.2f},{x_past[:,5].mean():.2f})",
                end="\t"
            )

        # Compute the new state updating the previous one
        noise_all = npr.randn(self.N, P) * dinamic_noise
        x_new = (self.A @ x_past.T).T + noise_all
        
        return x_new
    
    def update_hist_ref(self, im_hsv, x_global, Neff, beta=0.1):
        """
        Updates the reference histogram based on the current estimated bbox (x_global)
        and the reliability of the particle set (Neff).

        Parameters
        ----------
        im_hsv : ndarray
            Current image in HSV space (H and S channels).
        x_global : array-like
            State vector representing the estimated object [x, y, w, h, ...].
        Neff : float
            Effective number of particles (between 0 and 1).
        beta : float
            Base update rate (can be scaled by Neff).
        """
        # Reset reference if object is lost to avoid learning background
        if Neff < cfg.lost_obj_Neff_th:
            self.hist_ref = self.hist_init.copy()
            return

        # Extract histogram of the estimated bbox
        hist_candidate = self.get_particle_hist_(x_global, im_hsv)

        # Compute similarity with current reference
        sim_global = self.get_Battacharyya_(hist_candidate)

        # Update rate scales with Neff and similarity
        update_rate = np.clip((Neff / cfg.dinamic_Neff_th) * 2*beta, 0, beta)

        # Only update if similarity is reasonably high
        if cfg.DEBUG:
            print(f"sim of hist: {sim_global:.3f}", end=" | ")
        if sim_global > cfg.hist_update_th:
            if cfg.DEBUG:
                print(f"updated {update_rate:.3f}%", end=" | ")
            self.hist_ref = (1 - update_rate) * self.hist_ref + update_rate * hist_candidate
            
            # Anchor to initial histogram to prevent drifting
            self.hist_ref = (1 - cfg.anchor_weight) * self.hist_ref + cfg.anchor_weight * self.hist_init
            self.hist_ref /= self.hist_ref.sum() + 1e-10
    
    def compute_deep_embedding_single_(self, patch):
        """Helper to compute embedding for a single patch (used for reference)"""
        if patch.size == 0: return torch.zeros(1000).to(self.device)
        tensor = self.transform(patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.dl_model(tensor)
        return embedding # Shape (1, 1000)

    def compute_deep_likelihoods_batch_(self, particles, im_rgb):
        """
        Extracts all particle patches, stacks them into a batch, and runs the CNN once.
        """
        height, width, _ = im_rgb.shape
        batch_tensors = []
        valid_indices = []
        
        # 1. Extract and Preprocess Patches
        for i in range(self.N):
            x, y, w, h = particles[i, :4]
            
            limy = np.clip([np.ceil(y - 0.5 * h), np.floor(y + 0.5 * h)], 0, height).astype(int)
            limx = np.clip([np.ceil(x - 0.5 * w), np.floor(x + 0.5 * w)], 0, width).astype(int)
            
            patch = im_rgb[limy[0]:limy[1], limx[0]:limx[1], :]
            
            if patch.size > 0 and patch.shape[0] > 0 and patch.shape[1] > 0:
                try:
                    tensor = self.transform(patch)
                    batch_tensors.append(tensor)
                    valid_indices.append(i)
                except Exception:
                    pass # Skip malformed patches

        if not batch_tensors:
            return np.zeros(self.N)

        # 2. Stack into a single tensor (Batch Size, 3, H, W)
        batch_input = torch.stack(batch_tensors).to(self.device)
        
        # 3. Run Inference (One forward pass for N particles)
        with torch.no_grad():
            embeddings = self.dl_model(batch_input) # Shape (N_valid, 1000)
        
        # 4. Compute Cosine Similarity with Reference
        # self.dl_ref is (1, 1000), embeddings is (N_valid, 1000)
        # Cosine Sim = (A . B) / (|A|*|B|)
        # F.cosine_similarity computes similarity along dim 1
        sims = F.cosine_similarity(self.dl_ref, embeddings)
        
        # 5. Map back to weights array
        weights = np.zeros(self.N)
        sims_np = sims.cpu().numpy()
        
        # Clip negative similarities (DL embeddings can be orthogonal or opposite)
        sims_np = np.clip(sims_np, 0, 1.0)
        
        # Apply exponent alpha to sharpen distribution (like in Bhattacharyya)
        weights[valid_indices] = sims_np ** self.alpha
        
        return weights

    def compute_color_likelihoods_batch_(self, particles, im_hsv):
        """
        Computes color histogram likelihoods for all particles without MCMC.
        Used when DL mode is active.
        """
        weights = np.zeros(self.N)
        for i in range(self.N):
            hist = self.get_particle_hist_(particles[i], im_hsv)
            sim = self.get_Battacharyya_(hist)
            weights[i] = sim ** self.alpha
        return weights

    def weight_update_mcmc_(self, x_new, P, im_hsv):
        """
        Applies a Metropolis-Hastings MCMC move to refine particle states after
        propagation. Proposals are generated via Gaussian perturbations and
        accepted based on the Bhattacharyya likelihood ratio, improving the
        approximation of the posterior p(x_t | z_t).
        """
        n_accpeted = 0
        p_proposal_mean = 0
        p_curr_mean = 0
        # Propagate the particles with MCMC 
        # Particles loop (this cannot be paralelized)
        for i in range(self.N):
            # 1- Proposal of particle movment inverse proportional to the weight g=Gaussian
            # + weight - MCMC step | - weight + MCMC step
            w_norm = self.w[i] / (np.max(self.w) + 1e-10)     # Normalized version
            # w = self.w[i]
            scale = cfg.mcmc_expl_fact * (1-w_norm)
           
            # scale = np.clip(cfg.mcmc_expl_fact * (1 - self.w[i]), 0, cfg.mcmc_expl_fact)
            proposal = x_new[i] + npr.randn(P) * scale * self.Sigma
            
            # 2- Limit the proposal inisde the image
            proposal[0] = np.clip(proposal[0], 0, im_hsv.shape[1])
            proposal[1] = np.clip(proposal[1], 0, im_hsv.shape[0])
            proposal[2:4] = np.clip(proposal[2:4], 1, np.min(np.array(im_hsv.shape[0:2])))
            
            # 3- Get the candidate histogram
            hist_prop = self.get_particle_hist_(proposal, im_hsv)
            hist_curr = self.get_particle_hist_(x_new[i], im_hsv)
            
            # 4- get the proposal probability and original probability
            p_proposal = self.get_Battacharyya_(hist_prop) ** self.alpha
            p_current = self.get_Battacharyya_(hist_curr) ** self.alpha
            p_proposal_mean += p_proposal
            p_curr_mean += p_current

            # 5- Accpetance ratio α = p(x′)/p(xt​) ​(Metropolis-Hastings)
            accept_ratio = min(1.0, p_proposal / (p_current + 1e-10))

            # 6- Accept probability u ≤ α / u∼U(0,1) and update the weights
            if npr.rand() <= accept_ratio:
                x_new[i] = proposal  # Accept the proposal
                self.w[i] = p_proposal
                n_accpeted += 1
            else: # we keep the original particle
                self.w[i] = p_current

        p_proposal_mean /= self.N
        p_curr_mean /= self.N
        
        if cfg.DEBUG:
            print(f"Accpeted: ({n_accpeted}/{self.N})", end="")
    
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
                # Spatial Histogram: Split into Top and Bottom
                h_c, _, _ = candidate_reg.shape
                h_half = h_c // 2
                hist_top = computeMultiChannelHistogram(candidate_reg[:h_half, :, :], self.K)
                hist_bottom = computeMultiChannelHistogram(candidate_reg[h_half:, :, :], self.K)
                hist = np.concatenate((hist_top, hist_bottom)) / 2.0

        except Exception:
            hist = np.zeros_like(self.hist_ref)

        return hist

    def get_Battacharyya_(self, hist):
        return (np.sqrt(self.hist_ref * hist)).sum()