# Math utility functions file
"""
Ontogeny OMR (Optomotor Response) script for math utility functions

In this script you can find mathematical helper functions for geometric
calculations, signal filtering, and trajectory analysis.
"""

import numpy as np
import smallestenclosingcircle
from scipy import signal
from scipy.ndimage import maximum_filter1d
import config  # Import constants

def compute_distance2border(x, y, circle):
    """
    Calculates Euclidean distance from a point to the circle's edge.
    
    Args:
        x (float): Point x-coordinate.
        y (float): Point y-coordinate.
        circle (tuple): Circle definition (center_x, center_y, radius).
        
    Returns:
        float: Distance to border (positive = inside, negative = outside).
    """
    dist2center = np.sqrt((x - circle[0])**2 + (y - circle[1])**2)
    return circle[2] - dist2center

def compute_arena_circle(x_mm, y_mm):
    """
    Computes the smallest enclosing circle for the fish trajectory.
    
    Args:
        x_mm (np.array): X positions in mm.
        y_mm (np.array): Y positions in mm.
        
    Returns:
        tuple: (center_x, center_y, radius) or None if insufficient points.
    """
    # Subsample points for efficiency
    points = [(x_mm[i], y_mm[i]) 
              for i in range(0, len(x_mm), config.BORDER_SAMPLING_STEP) 
              if not np.isnan(x_mm[i])]
    
    if len(points) > config.MIN_CIRCLE_POINTS:
        return smallestenclosingcircle.make_circle(points)
    return None

def max_filter1d_valid(a, W):
    """
    Applies a 1D maximum filter with reflection handling.
    Used for envelope enhancement in bout detection.
    
    Args:
        a (np.array): Input array.
        W (int): Window size.
        
    Returns:
        np.array: Filtered array.
    """
    return maximum_filter1d(a, size=W, mode='reflect')

def find_outlier_trajectory(df, pix_size, smoothing_win=10):
    """
    Flags tracking errors based on physically impossible velocity thresholds.
    
    Args:
        df (pd.DataFrame): Data containing 'x_pos', 'y_pos', and 'body_angle'.
        pix_size (float): Resolution in microns/pixel (specific to each fish).
        smoothing_win (int): Window size to smooth error flags.
        
    Returns:
        np.array: Array of 1.0 (error) and 0.0 (valid) for each frame.
    """
    # Calculate dynamic thresholds based on Config constants
    # 1. Angular limit: (deg/ms * ms/s) / fps = deg/frame
    ang_thresh_per_frame = (config.MAX_ANG_VEL_DEG_PER_MS * config.MS_PER_SEC) / config.FPS
    
    # 2. Linear limit: (mm/s / (um/px / um/mm)) / fps = px/frame
    mm_per_pixel = pix_size / config.MICRONS_PER_MM
    lin_thresh_per_frame = (config.MAX_LIN_VEL_MM_PER_S / mm_per_pixel) / config.FPS

    # Calculate actual speeds
    angular_speed = 180 / np.pi * np.abs(np.diff(np.unwrap(df['body_angle'].values), prepend=0)) # degrees per frame
    linear_speed = np.sqrt(np.diff(df['x_pos'].values, prepend=0)**2 + np.diff(df['y_pos'].values, prepend=0)**2)
    
    # Identify outliers
    invalid_indices = np.union1d(
        np.where(angular_speed > ang_thresh_per_frame)[0],
        np.where(linear_speed > lin_thresh_per_frame)[0]
    )
    
    fail_mask = np.zeros(len(df))
    fail_mask[invalid_indices] = 1
    
    # Smooth the result (if a frame is bad, neighbors likely are too)
    smoothed_fail = signal.lfilter(np.ones(smoothing_win), 1, fail_mask) > 0
    return smoothed_fail.astype(float)