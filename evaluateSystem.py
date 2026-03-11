#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:12:37 2022

@author: ivan
"""

import numpy as np
import argparse
import sys
import os
from object_tracking import object_tracking
from multiprocessing import Pool
from functools import partial
import pdb
import config as cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a system for visual object tracking')
    parser.add_argument('--repetitions', dest='repetitions',
                        help='number of repetitions for each video. Larger values improve stability bu take longer (default 4)',
                        default=4, type=int)
    parser.add_argument('--N', dest='N',  
                       help='numParticles to be used in the filter (default N=300)',
                       type=int, default=300)
    
    # New arguments for parameter tuning
    parser.add_argument('--sigma', nargs=8, type=float,
                        help='Standard deviation of noise for state vector [x,y,w,h,vx,vy,vw,vh]',
                        default=cfg.std_noise)
    parser.add_argument('--alpha', type=float,
                        help='Exponent for Bhattacharyya similarity',
                        default=cfg.alpha)
    parser.add_argument('--speed_noise_factor', type=float,
                        help='Factor to scale positional noise with speed',
                        default=cfg.speed_noise_factor)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # Override config values with command-line arguments
    cfg.std_noise = np.array(args.sigma)
    cfg.alpha = args.alpha
    cfg.speed_noise_factor = args.speed_noise_factor
    
    videos={'Basketball','Biker','Bolt','Skating'};
    # videos={'Biker'}
    #As the particle filtering is stichastic, several repetitions will provide
    #different values. we compute several and average to get more stable
    #results.
    numRepetitions=args.repetitions;

    JI=np.zeros((len(videos),numRepetitions));
    time_per_frame=np.zeros((len(videos),numRepetitions));
    for v,video in enumerate(videos):
        print('============================');
        print('Processing video %s'%video);
        print('============================');
        aux_input=np.random.randint(0,1000,(numRepetitions,))
        pool = Pool(processes=np.minimum(os.cpu_count(),numRepetitions))
        
        results=pool.map(partial(object_tracking, video,args.N,False),aux_input)
        results=np.array(results)
        JI[v,:]=results[:,0]
        time_per_frame[v,:]=results[:,1]
        pool.close()
        pool.join()      
        
    print('============================');
    print('Summary of Final results')
    print('============================');
    for v,video in enumerate(videos):
        print('Results for video %s JI=%f and avg_time_per_frame=%f secs'%(video,JI[v,:].mean(),time_per_frame[v,:].mean()));
    print('============================');
    print('Total average Results are JI=%f'%JI.mean());
    print('============================');