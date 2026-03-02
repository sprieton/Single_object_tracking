#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:34:54 2022

@author: ivan
"""
import numpy as np
from skimage.color import gray2rgb
from skimage.util import img_as_ubyte
import cv2
import pdb

def showBB(im, bbs):
    if im.ndim == 2:
        im = gray2rgb(im)
    im = img_as_ubyte(im)
    
    # BGR format colors for OpenCV (or RGB depending on your imshow logic)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), 
              (0, 255, 255), (255, 0, 255), (255, 128, 0), (0, 255, 128), 
              (128, 255, 0), (128, 128, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)]
    
    for b in range(bbs.shape[0]):
        bb = np.round(bbs[b, :]).astype(int)
        
        # bb is [x, y, width, height]
        x1, y1 = bb[0], bb[1]
        x2, y2 = bb[0] + bb[2], bb[1] + bb[3]
        
        # cv2.rectangle handles image boundaries automatically, no try/except needed
        color = colors[b % len(colors)]
        cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        
    return im

def showParticles(im, x):
    """
    Function that shows particles as points overlaid in the image (we do not consider width, height and dynamic elements in the state)
          im = showParticles(im,x)     
    Parameters:
           - im: the image
           - x: The matrix with the state respresented by the particles
    Output:
            - im: the image with overlaid particles
    """
    if im.ndim == 2:
        im = gray2rgb(im)
    im = img_as_ubyte(im)

    # Draw particles directly as small circles (lightning fast in OpenCV)
    for i in range(x.shape[0]):
        pt = (int(x[i, 0]), int(x[i, 1]))
        cv2.circle(im, pt, 1, (255, 255, 255), -1)
        
    return im
