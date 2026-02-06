# Processing file
"""
Ontogeny OMR (Optomotor Response) script for main analysis-related logic

In this script you can find the core processing logic for individual
recordings. It coordinates loading data, running behavioral analysis,
merging datasets, and saving the final output.
"""

import os
import pandas as pd
import numpy as np
import config
import math_utils
import io_utils

def compute_tail_signals(tail_angle_array):
    """
    Computes cumulative and smoothed tail angles for logging purposes.
    These metrics are saved to the file regardless of the detection method used.
    
    Args:
        tail_angle_array (np.array): Matrix of tail angles (frames x segments).
        
    Returns:
        tuple: (cumul_tail, smooth_tail)
    """
    from scipy import signal # Import locally to avoid cluttering global namespace
    
    tail_data = tail_angle_array.copy()
    tail_data[np.isnan(tail_data)] = 0
    
    # 1. Cumulative Sum (Spatial)
    # Sums angles along the tail to get the total curvature
    cumul_tail = np.cumsum(tail_data, axis=1)
    
    # 2. Smoothing (Savgol filter)
    # Smooths the signal over time to remove high-frequency noise
    smooth_tail = np.zeros_like(cumul_tail)
    for i in range(cumul_tail.shape[1]):
        smooth_tail[:, i] = signal.savgol_filter(cumul_tail[:, i], window_length=11, polyorder=2)
        
    return cumul_tail, smooth_tail

def detect_bouts_manual(smooth_tail, fs):
    """
    Original custom algorithm: Detects bouts using tail angle change (motion energy) 
    envelope thresholding.
    
    Args:
        smooth_tail (np.array): Smoothed cumulative tail angles.
        fs (float): Sampling frequency.
        
    Returns:
        tuple: (start_indices, end_indices)
    """
    from scipy import signal
    
    # Calculate tail angle change (motion energy)
    diff_tail = np.diff(smooth_tail, axis=0, prepend=0)
    motion_signal = np.sum(np.abs(diff_tail), axis=1)
    
    # Bout envelope - filter signal to smooth tail beats
    boxcar_size = 10
    motion_envelope = signal.convolve(motion_signal, np.ones(boxcar_size)/boxcar_size, mode='same')
    
    # Enhancement using helper function from math_utils
    max_filt = math_utils.max_filter1d_valid(motion_envelope, W=20)
    min_filt = -math_utils.max_filter1d_valid(-motion_envelope, W=400)
    enhanced_signal = max_filt - min_filt
    
    # Thresholding
    bout_thresh = 0.1
    is_bout = (enhanced_signal > bout_thresh).astype(int)
    diff_bout = np.diff(is_bout, prepend=0)
    starts = np.where(diff_bout == 1)[0]
    ends = np.where(diff_bout == -1)[0]
    
    # Cleanup edge cases
    if len(starts) > len(ends): starts = starts[:-1]
    if len(ends) > len(starts): ends = ends[1:]
    
    # Filter by duration and amplitude
    min_length = 40
    min_amp = 0.25
    valid_starts = []
    valid_ends = []
    
    for s, e in zip(starts, ends):
        duration = e - s
        if e <= len(enhanced_signal):
            amplitude = np.max(enhanced_signal[s:e])
            if duration > min_length and amplitude > min_amp:
                valid_starts.append(s)
                valid_ends.append(e)
                
    return np.array(valid_starts), np.array(valid_ends)

def analyze_behavior(tail_angle_array, fs):
    """
    Main wrapper that computes signals and runs the selected detection algorithm.
    
    Args:
        tail_angle_array (np.array): Raw tail angle data.
        fs (float): Sampling frequency.
        
    Returns:
        dict: Processed behavioral data (starts, ends, signals).
    """
    # 1. Always compute standard signals
    cumul_tail, smooth_tail = compute_tail_signals(tail_angle_array)
    
    # 2. Run Detection based on Config Flag
    if config.MEGABOUTS_FLAG and config.MEGABOUTS_AVAILABLE:
        print("  [Behavior] Using MEGABOUTS for segmentation.")
        try:
            # Import megabouts only if needed
            from megabouts.tracking_data import TailTrackingData
            from megabouts.segmentation import TailSegmentation
            from megabouts.config.segmentation_config import TailSegmentationConfig
            
            seg_cfg = TailSegmentationConfig(fps=fs, min_bout_duration=40, bout_thresh=0.1)
            tracking_data = TailTrackingData.from_array(tail_angle_array, fps=fs)
            segmenter = TailSegmentation(tracking_data, seg_cfg)
            results = segmenter.run()
            
            starts = np.array([b.start for b in results.bouts])
            ends = np.array([b.end for b in results.bouts])
            
        except Exception as e:
            print(f"  [Error] Megabouts failed ({e}). Falling back to manual.")
            starts, ends = detect_bouts_manual(smooth_tail, fs)
    else:
        print("  [Behavior] Using Manual Thresholding for segmentation.")
        starts, ends = detect_bouts_manual(smooth_tail, fs)
        
    return {
        'cumul_tail': cumul_tail,
        'smooth_tail': smooth_tail,
        'bout_starts': starts,
        'bout_ends': ends
    }

def process_recording(row):
    """
    Runs conversion, analysis, and merging pipeline for a single recording.
    
    Args:
        row (pd.Series): Metadata for one experiment.
        
    Returns:
        bool: Success status.
    """
    folder = row['fish_folder']
    base_name = row['fish_filename'].replace('.txt', '')
    
    # Define file paths
    raw_cam_mat = os.path.join(folder, base_name + '.mat')
    pkl_cam = os.path.join(folder, base_name + '.pickle')
    stim_name = row['fish_stimlog_filename']
    
    if not stim_name: return False
        
    raw_stim_mat = os.path.join(folder, stim_name + '.mat')
    pkl_stim = os.path.join(folder, stim_name + '.pickle')
    final_merged = os.path.join(folder, base_name + '_MergedLog.pickle')

    print(f"Processing {base_name}...")

    # --- 1. Convert Stimulus Log ---
    if not os.path.exists(pkl_stim):
        try:
            df_stim = io_utils.load_stim_log(raw_stim_mat)
            rename_map = {
                'iGlobalTime': 'i_global_time', 'iTimeDelta': 'i_time_delta', 
                'Id': 'cam_frame', 'TimeCam': 'time_cam_shader', 
                'xPos': 'x_pos_shader', 'yPos': 'y_pos_shader', 
                'FOrient': 'fish_orientation_shader', 
                'Orientation': 'grating_orientation', 'Speed_mm': 'grating_speed'
            }
            df_stim.rename(columns=rename_map, inplace=True)
            df_stim.to_pickle(pkl_stim, compression='infer')
        except Exception as e:
            print(f"  [Error] StimLog conversion: {e}")
            return False
    else:
        df_stim = pd.read_pickle(pkl_stim)

    # --- 2. Convert Camera Log ---
    if not os.path.exists(pkl_cam):
        try:
            df_cam = io_utils.load_cam_log(raw_cam_mat)
            
            # Load the schema from io_utils (guarantees correct naming)
            col_names = io_utils.get_camera_column_names()
            
            width = df_cam.shape[1]
            expected_width = len(col_names)
            
            # SAFETY CHECK: Alignment
            if width == expected_width:
                df_cam.columns = col_names
            else:
                # Warning: If dimensions mismatch, assign what fits to avoid crash
                print(f"  [WARNING] Column Mismatch! Raw matrix has {width} cols, expected {expected_width}.")
                print(f"  -> Data might be misaligned. Assigning first {width} names.")
                df_cam.columns = col_names[:width]

            # Ensure unique names (handles any duplicate columns in the raw data)
            df_cam = io_utils.addindex2identicalcolumnsname(df_cam)
            df_cam.to_pickle(pkl_cam, compression='infer')
        except Exception as e:
            print(f"  [Error] CamLog conversion: {e}")
            return False
    else:
        df_cam = pd.read_pickle(pkl_cam)

    # --- 3. Behavioral Analysis ---
    pix_size = row['fish_resolution']
    x_mm = df_cam['x_pos'] * pix_size
    y_mm = df_cam['y_pos'] * pix_size
    
    # Distance to Border
    arena_circle = math_utils.compute_arena_circle(x_mm, y_mm)
    if arena_circle:
        dist_microns = math_utils.compute_distance2border(x_mm, y_mm, arena_circle)
        df_cam['distance2border_mm'] = dist_microns / config.MICRONS_PER_MM
    else:
        df_cam['distance2border_mm'] = np.nan

    # Tracking Failures
    fail_mask = math_utils.find_outlier_trajectory(df_cam, pix_size)
    df_cam['fail_body_tracking'] = pd.Series(fail_mask, index=df_cam.index)

    # Detect Bouts
    # Grab all columns that contain 'tail_angle' in their name
    tail_cols = [c for c in df_cam.columns if 'tail_angle' in c]
    bout_results = analyze_behavior(df_cam[tail_cols].values, config.FPS)
    
    # --- 4. Merge Data ---
    cam_indexed = df_cam.set_index('frame_number') # set the frame number and the index
    stim_subset = df_stim[['cam_frame', 'grating_orientation', 'grating_speed']]
    stim_subset = stim_subset.drop_duplicates(subset='cam_frame').set_index('cam_frame')
    
    # Left join to keep all camera frames
    merged = cam_indexed.join(stim_subset, how='left')
    merged['grating_orientation'] = merged['grating_orientation'].fillna(method='ffill')
    merged['grating_speed'] = merged['grating_speed'].fillna(method='ffill')
    
    # Append Bout Signals
    for i in range(bout_results['cumul_tail'].shape[1]):
        merged[f'cumul_tail_angle.{i}'] = bout_results['cumul_tail'][:, i]
        merged[f'smooth_cumul_tail_angle.{i}'] = bout_results['smooth_tail'][:, i]

    # Append Bout Indices
    bout_array = np.zeros(len(merged))
    frame_lookup = df_cam['frame_number'].values
    merged['id_bout_start_ind'] = np.nan
    merged['id_bout_end_ind'] = np.nan
    
    # Map array indices back to real frame numbers
    for k, (s, e) in enumerate(zip(bout_results['bout_starts'], bout_results['bout_ends'])):
        if s < len(frame_lookup) and e < len(frame_lookup):
            f_s, f_e = frame_lookup[s], frame_lookup[e]
            
            if f_s in merged.index: merged.at[f_s, 'id_bout_start_ind'] = k
            if f_e in merged.index: merged.at[f_e, 'id_bout_end_ind'] = k
            
            # Mark active duration (using k+1 so 0=rest)
            if e < len(bout_array): bout_array[s:e] = k + 1

    merged['tail_active'] = bout_array
    
    # --- 5. Cleanup & Save ---
    # Drops raw vectors and raw tail values (keeping only angles)
    # Uses tail_value_{i} format matching io_utils
    drop_cols = ['fish_blob_val', 'max_val', 'x_body_vect', 'y_body_vect'] + \
                [f'tail_value_{i}' for i in range(10)]
                
    merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)
    merged.to_pickle(final_merged, compression='infer')
    
    print(f"  -> Saved: {final_merged}")
    return True