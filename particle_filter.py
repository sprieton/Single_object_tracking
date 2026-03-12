#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:06:37 2022

@author: ivan
"""

import numpy as np
import cv2
from skimage.color import rgb2hsv
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

def computeMultiChannelHistogram(im, K, mask=None):
    """
    Computes a normalized joint histogram for an image with a variable number of channels.
    Assumes that the input values of 'im' are normalized in the range [0.0, 1.0].
    
    Parameters:
    - im: 3-dimensional NumPy array (Height, Width, Channels).
    - K: Number of bins per channel.
    - mask: Optional 2D boolean/binary array. If provided, only pixels where mask > 0 are used.
    
    Returns:
    - hist: 1D vector containing the normalized joint histogram of size K^Channels.
    """
    # In case we have a 2D image with just one channel 
    if im.ndim == 2:
        im = im[:, :, np.newaxis]
        
    h, w, c = im.shape
    
    # 1. Vectorize the image to have a list of pixels (H*W, C)
    if mask is not None:
        # Use only pixels within the mask
        im_flat = im[mask.astype(bool)]
    else:
        im_flat = np.reshape(im, (h * w, c))
    
    # 2. Quantize the values. 
    # Subtract 1e-30 to ensure the maximum value (1.0) falls into bin K-1 and doesn't go out of bounds.
    r = np.floor((im_flat - 1e-30) * K).astype(int)
    
    # For safety, ensure no index goes out of the [0, K-1] boundaries
    r = np.clip(r, 0, K - 1)
    
    # 3. Compute the linear index for np.bincount (base K to base 10 conversion)
    rlin = np.zeros(im_flat.shape[0], dtype=int)
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

        # Compute descriptor using the configured strategy (Spatial or Ellipse)
        self.hist_ref = self._compute_histogram_descriptor(objim_hsv)
        
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
        self.vel_history = []       # save the velocities to make a mean

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

        # 4- Determine speed factor and adaptive t-Student degrees of freedom (variance of the t-Student)
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
        scale_perp = dinamic_noise[:2] * speed_factor * 0.5  # smaller lateral component

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
           
            # 2- Create the proposal using the scales of exploration
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

        p_proposal_mean /= self.N
        p_curr_mean /= self.N
        
        if cfg.DEBUG:
            print(f"Accpeted: ({n_accepted}/{self.N})", end="")
    
    def _compute_histogram_descriptor(self, patch_hsv):
        """
        Computes the histogram descriptor based on the configuration (Spatial Split or Ellipse).
        Helper method to avoid code duplication.
        """
        h_c, w_c, _ = patch_hsv.shape

        if cfg.observation_model == 'ellipse_hist':
            # Calculate scaling factor to match the desired area ratio
            # Area_ellipse = pi * (s*W/2) * (s*H/2) = s^2 * (pi/4) * Area_box
            # We want Area_ellipse = ratio * Area_box
            # s = 2 * sqrt(ratio / pi)
            s = 2 * np.sqrt(cfg.ellipse_area_ratio / np.pi)
            
            center = (w_c // 2, h_c // 2)
            axes = (int(s * w_c / 2), int(s * h_c / 2))
            
            mask_in = np.zeros((h_c, w_c), dtype=np.uint8)
            cv2.ellipse(mask_in, center, axes, 0, 0, 360, 1, -1)
            mask_out = 1 - mask_in
            
            hist_in = computeMultiChannelHistogram(patch_hsv, self.K, mask=mask_in)
            hist_out = computeMultiChannelHistogram(patch_hsv, self.K, mask=mask_out)
            return np.concatenate((hist_in, hist_out)) / 2.0
        
        else: # Default: 'spatial_hist' (Top / Bottom split)
            h_half = h_c // 2
            hist_top = computeMultiChannelHistogram(patch_hsv[:h_half, :, :], self.K)
            hist_bottom = computeMultiChannelHistogram(patch_hsv[h_half:, :, :], self.K)
            return np.concatenate((hist_top, hist_bottom)) / 2.0

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
                hist = self._compute_histogram_descriptor(candidate_reg)

        except Exception:
            hist = np.zeros_like(self.hist_ref)

        return hist

    def get_Battacharyya_(self, hist):
        return (np.sqrt(self.hist_ref * hist)).sum()