# Lab Session 3 – Visual Object Tracking  
## Computer Vision  
Master in Machine Learning for Health (2025/2026)

## Practice Overview

This lab focuses on implementing a **Particle Filter for visual object tracking**. The objective is to track an object across video frames by estimating its position, size, and motion over time using a probabilistic approach.

The tracker represents the object state using a vector that includes:

- Position (x, y)
- Bounding box size (width and height)
- Velocity components for both position and size

Instead of estimating a single solution, the particle filter maintains a set of hypotheses (particles). Each particle represents a possible state of the object. Over time, these particles are updated and weighted according to how well they explain the current image observation.

## What the Practice Consists Of

During this lab, students:

1. Implement a particle filter for object tracking.
2. Define a state transition model to predict object motion between frames.
3. Define an observation model to evaluate how well each particle matches the object appearance.
4. Perform particle weighting and resampling.
5. Analyze tracking performance under different parameter settings.

## Methodology

At each frame, the algorithm follows these steps:

1. **Prediction** – Particles are propagated using a motion model with added Gaussian noise.
2. **Evaluation** – Each particle is assigned a weight based on how well it matches the observed object in the image.
3. **Resampling** – Particles with higher weights are more likely to be kept, while low-weight particles are discarded.
4. **Estimation** – The final object state is computed from the weighted particle set.

## Learning Objectives

By completing this practice, students will:

- Understand the principles of Bayesian filtering.
- Learn how particle filters approximate probability distributions.
- Gain experience in probabilistic visual tracking.
- Evaluate robustness to motion, scale changes, and noise.

## Expected Outcome

At the end of the lab, students should have a working object tracker capable of following a target in a video sequence using a particle filtering framework.
