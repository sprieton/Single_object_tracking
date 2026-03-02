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

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a system for visual object tracking')
    parser.add_argument('--repetitions', dest='repetitions',
                        help='number of repetitions for each video. Larger values improve stability bu take longer (default 4)',
                        default=4, type=int)
    parser.add_argument('--N', dest='N',  
                       help='numParticles to be used in the filter (default N=100)',
                       type=int, default=100)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    
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