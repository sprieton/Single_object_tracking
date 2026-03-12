#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:04:29 2022

@author: ivan
"""
import numpy as np

def computeJI(gtbox, predbox):
    # Calculate right and bottom edges WITHOUT modifying the original arrays
    gt_right, gt_bottom = gtbox[0] + gtbox[2], gtbox[1] + gtbox[3]
    pred_right, pred_bottom = predbox[0] + predbox[2], predbox[1] + predbox[3]
    
    x_left = np.maximum(gtbox[0], predbox[0])
    y_top = np.maximum(gtbox[1], predbox[1])
    x_right = np.minimum(gt_right, pred_right)
    y_bottom = np.minimum(gt_bottom, pred_bottom)

    # if they do not intersect
    if x_right < x_left or y_bottom < y_top:
        JI = 0.0
    else:
        # Intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Union areas (width * height)
        gt_area = gtbox[2] * gtbox[3]
        pred_area = predbox[2] * predbox[3]
        JI = intersection_area / (gt_area + pred_area - intersection_area)
        
    return JI