#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:52:06 2022

@author: ivan
"""

import numpy as np
import cv2
import argparse
import sys
import matplotlib.pyplot as plt
import pdb
from scipy.io import loadmat
from particle_filter import particle_filter
from visualization import showBB, showParticles
from metrics import computeJI
import time                       

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Compute the depth map and 3D reconstruction from an image stereoscopic pair')
    parser.add_argument('video', metavar='video', type=str, nargs=1,
                    help='the name of the video')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='if set, visual results are shown (default False)')
    parser.add_argument('--N', dest='N',  
                        help='numParticles to be used in the filter (default N=100)',
                        type=int, default=100)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def object_tracking(video, N, VERBOSE, seed=1):
    
    
    videoPath='./videos/' + video
    labelPath='./videos/%s.mat'%video
    np.random.seed(seed)
    aux = loadmat(labelPath)
    labels=aux['labels'].astype(int)
      
    numFrames=labels.shape[0]
    numFrames=np.minimum(numFrames,200);
    fr_path = '%s/0001.jpg' % videoPath
    im = cv2.imread(fr_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # Keep RGB format
    
    #We initialize JI
    JI=np.zeros((numFrames,));
    JI[0]=1.0;


    #We initialize the object tracker using the ground truth and the configuration in config.py
    tracker=particle_filter(im,labels[0,:],numParticles=N,step=1);
    if VERBOSE:
        #Show the tracker
        im=showBB(im,np.vstack((labels[0,:], tracker.bbox)));
        #Show the particles
        im=showParticles(im,tracker.x);
        cv2.imshow('Tracking', im[:,:,::-1])
        cv2.waitKey(1) 
    tic = time.time()
    #Loop of frames
    for n in range(1,numFrames):
        #Read the frame
        fr_path='%s/%04d.jpg'%(videoPath,n+1)
        im = cv2.imread(fr_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #Update the particle filter
        tracker.update(im);
        if VERBOSE:
            #Show the tracker
            im=showBB(im,np.vstack((labels[n,:], tracker.bbox)));
            #Show the particles
            im=showParticles(im,tracker.x);
            cv2.imshow('Tracking', im[:,:,::-1])
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #Compute the Jacard-Index
        JI[n]=computeJI(labels[n,:4],tracker.bbox[:4]);
        print(f"Jaccard:{JI[n]}")
        if n%50==0:
            print('Frame %d/%d average JI %f'%(n,numFrames,JI[:n+1].mean()));
        
    toc = time.time()
    avg_time=(toc-tic)/numFrames
    avg_JI=JI.mean()
    
    plt.figure(1)
    plt.plot(JI)
    return avg_JI,avg_time
    
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    results=object_tracking(args.video[0],args.N,args.verbose)
    print('Results on video {0} are JI: {1:.3f} and avg computation time {2:.3f} secs'.format(args.video[0],results[0],results[1]))