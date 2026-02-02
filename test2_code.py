import os
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import math
import random
import warnings
import smallestenclosingcircle  # pip install smallestenclosingcircle
from scipy import signal
from scipy.ndimage import maximum_filter1d

# --- CONFIGURATION FLAGS ---
MEGABOUTS_FLAG = False  # Set to True to use the megabouts library

# Try importing megabouts if flag is set
if MEGABOUTS_FLAG:
    try:
        from megabouts.tracking_data import TailTrackingData
        from megabouts.segmentation import TailSegmentation
        from megabouts.config.segmentation_config import TailSegmentationConfig
    except ImportError:
        print("[Warning] MEGABOUTS_FLAG is True but library not found. Falling back to manual detection.")
        MEGABOUTS_FLAG = False

warnings.filterwarnings('ignore')

# ==============================================================================
# SECTION 1: HELPER ALGORITHMS (Geometry & Tracking)
# ==============================================================================

def compute_distance2border(x, y, circle):
    """Calculates Euclidean distance from a point to the circle's edge."""
    dist2center = np.sqrt((x - circle[0])**2 + (y - circle[1])**2)
    return circle[2] - dist2center

def find_outlier_trajectory(df, max_angular_speed, max_speed, smoothing_win=10):
    """Flags tracking errors based on impossible velocity thresholds."""
    angular_speed = 180 / np.pi * np.abs(np.diff(np.unwrap(df['body_angle'].values), prepend=0))
    linear_speed = np.sqrt(np.diff(df['x_pos'].values, prepend=0)**2 + np.diff(df['y_pos'].values, prepend=0)**2)
    
    invalid_indices = np.union1d(
        np.where(angular_speed > max_angular_speed)[0],
        np.where(linear_speed > max_speed)[0]
    )
    
    fail_mask = np.zeros(len(df))
    fail_mask[invalid_indices] = 1
    smoothed_fail = signal.lfilter(np.ones(smoothing_win), 1, fail_mask) > 0
    return smoothed_fail.astype(float)

def max_filter1d_valid(a, W):
    """Applies a 1D maximum filter with reflection handling."""
    return maximum_filter1d(a, size=W, mode='reflect')

def addindex2identicalcolumnsname(df):
    """Renames duplicate DataFrame columns by appending a counter."""
    cols = pd.Series(df.columns)
    for dup in df.columns[df.columns.duplicated()].unique():
        cols[df.columns.get_loc(dup)] = [dup + '.' + str(d_idx) for d_idx in range(df.columns.get_loc(dup).sum())]
    df.columns = cols
    return df

# ==============================================================================
# SECTION 2: BEHAVIORAL ALGORITHMS (Signal Processing & Bouts)
# ==============================================================================

def compute_tail_signals(tail_angle_array):
    """
    Computes cumulative and smoothed tail angles for logging purposes.
    These metrics are saved to the file regardless of the detection method used.
    """
    tail_data = tail_angle_array.copy()
    tail_data[np.isnan(tail_data)] = 0
    
    # 1. Cumulative Sum (Spatial)
    cumul_tail = np.cumsum(tail_data, axis=1)
    
    # 2. Smoothing (Savgol filter)
    smooth_tail = np.zeros_like(cumul_tail)
    for i in range(cumul_tail.shape[1]):
        smooth_tail[:, i] = signal.savgol_filter(cumul_tail[:, i], window_length=11, polyorder=2)
        
    return cumul_tail, smooth_tail

def detect_bouts_manual(smooth_tail, fs=700):
    """
    Original custom algorithm: Detects bouts using motion energy envelope thresholding.
    """
    # Calculate motion energy
    diff_tail = np.diff(smooth_tail, axis=0, prepend=0)
    motion_signal = np.sum(np.abs(diff_tail), axis=1)
    
    # Envelope
    boxcar_size = 10
    motion_envelope = signal.convolve(motion_signal, np.ones(boxcar_size)/boxcar_size, mode='same')
    
    # Enhancement
    max_filt = max_filter1d_valid(motion_envelope, size=20)
    min_filt = -max_filter1d_valid(-motion_envelope, size=400)
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
    
    # Filters
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

def detect_bouts_megabouts(tail_angle_array, fs=700):
    """
    Uses the MEGABOUTS library to detect bouts.
    """
    # Create configuration
    seg_cfg = TailSegmentationConfig(
        fps=fs,
        min_bout_duration=40, # Matches manual min_length
        bout_thresh=0.1       # Matches manual bout_thresh
    )
    
    # Initialize data object
    # megabouts expects (n_frames, n_segments)
    tracking_data = TailTrackingData.from_array(tail_angle_array, fps=fs)
    
    # Run segmentation
    segmenter = TailSegmentation(tracking_data, seg_cfg)
    results = segmenter.run()
    
    # Extract indices
    # megabouts returns Bout objects with .start and .end attributes
    starts = [b.start for b in results.bouts]
    ends = [b.end for b in results.bouts]
    
    return np.array(starts), np.array(ends)

def analyze_behavior(tail_angle_array, fs=700, use_megabouts=False):
    """
    Main wrapper that computes signals and runs the selected detection algorithm.
    """
    # 1. Always compute standard signals for the log file
    cumul_tail, smooth_tail = compute_tail_signals(tail_angle_array)
    
    # 2. Run Detection
    if use_megabouts:
        print("  [Behavior] Using MEGABOUTS for segmentation.")
        starts, ends = detect_bouts_megabouts(tail_angle_array, fs)
    else:
        print("  [Behavior] Using Manual Thresholding for segmentation.")
        starts, ends = detect_bouts_manual(smooth_tail, fs)
        
    return {
        'cumul_tail': cumul_tail,
        'smooth_tail': smooth_tail,
        'bout_starts': starts,
        'bout_ends': ends
    }

# ==============================================================================
# SECTION 3: METADATA PARSING
# ==============================================================================

def parse_filename_metadata(filename, folder_path):
    """Extracts experimental parameters from the filename string."""
    clean_name = filename.replace('.txt', '').replace('.mat', '')
    parts = clean_name.split('_')
    
    meta = {
        'fish_filename': filename,
        'fish_folder': folder_path,
        'fish_resolution': 78.0, 
        'fish_circle': 66.0,
        'fish_strain': 'Unknown'
    }
    
    try:
        rig_raw = parts[-1]
        meta['fish_rig'] = ''.join([c for c in rig_raw if not c.isdigit()])
        meta['fish_circle'] = float(parts[-2])
        meta['fish_resolution'] = float(parts[-3])
        meta['fish_timeperiod'] = int(parts[-4].replace('P', ''))
        meta['fish_age'] = int(parts[-5].replace('dpf', ''))
        meta['fish_clutch'] = int(parts[-6].replace('C', ''))
        meta['fish_tank'] = int(parts[-7].replace('Tank', ''))
        meta['fish_strain'] = parts[-8]
        meta['fish_date'] = "_".join(parts[-11:-8])
    except (IndexError, ValueError):
        pass
        
    return meta

def generate_database(root_folder):
    """Scans directory tree to build a table of all fish experiments."""
    records = []
    print(f"Scanning {root_folder} for fish experiments...")
    
    for root, dirs, files in os.walk(root_folder):
        txt_files = [f for f in files if f.endswith('000.txt') and f.startswith('OMR_Ontogeny_VOL_')]
        
        for f in txt_files:
            entry = parse_filename_metadata(f, root)
            stim_files = [sf for sf in os.listdir(root) if 'stimlog' in sf and sf.endswith('.mat')]
            entry['fish_stimlog_filename'] = stim_files[0].replace('.mat', '') if stim_files else None
            
            base_name = f.replace('.txt', '')
            merged_path = os.path.join(root, base_name + '_MergedLog.pickle')
            entry['saving_flag'] = os.path.exists(merged_path)
            records.append(entry)
            
    return pd.DataFrame(records)

# ==============================================================================
# SECTION 4: LOG CONVERSION & MERGING
# ==============================================================================

def process_recording(row):
    """Runs conversion, analysis, and merging pipeline for a single recording."""
    folder = row['fish_folder']
    base_name = row['fish_filename'].replace('.txt', '')
    
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
            mat = sio.loadmat(raw_stim_mat, struct_as_record=False, squeeze_me=True)
            stim_data = mat['StimLog']
            data_dict = {}
            length = stim_data.Length
            
            for i in range(len(stim_data.Values)):
                name = stim_data.Names[i]
                vals = stim_data.Values[i].values
                pad_len = length - len(vals)
                padded_vals = np.pad(vals, (pad_len, 0), mode='constant', constant_values=np.nan)
                data_dict[name] = padded_vals
            
            df_stim = pd.DataFrame(data_dict)
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
            with h5py.File(raw_cam_mat, 'r') as f:
                cam_data = f['a'][()].T 
                
            col_names = [
                'frame_number', 'x_pos', 'y_pos', 'x_body_vect', 'y_body_vect', 'body_angle',
                'max_val', 'fish_blob_val', 'mid_eye_x', 'mid_eye_y', 'hour', 'minute',
                'stim_number', 'cum_sum_tail'
            ] + [f'tail_value.{i}' for i in range(10)] + ['body_angle_origin'] + \
                [f'tail_angle.{i}' for i in range(10)] + ['first_or_not', 'timing', 'lag']
            
            width = cam_data.shape[1]
            if width == len(col_names):
                df_cam = pd.DataFrame(cam_data, columns=col_names)
            else:
                df_cam = pd.DataFrame(cam_data)
                df_cam.columns = col_names[:width] 

            df_cam = addindex2identicalcolumnsname(df_cam)
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
    points = [(x_mm[i], y_mm[i]) for i in range(0, len(x_mm), 100) if not np.isnan(x_mm[i])]
    if len(points) > 5:
        arena_circle = smallestenclosingcircle.make_circle(points) 
        dist_mm = compute_distance2border(x_mm, y_mm, arena_circle)
        df_cam['distance2border_mm'] = dist_mm / 1000.0 
    else:
        df_cam['distance2border_mm'] = np.nan

    # Tracking Failures
    fps = 700.0
    fail_mask = find_outlier_trajectory(df_cam, (40.0 * 1000)/fps, (100.0 / (pix_size/1000.0))/fps)
    df_cam['fail_body_tracking'] = pd.Series(fail_mask, index=df_cam.index)

    # Detect Bouts (Conditional Logic)
    tail_cols = [c for c in df_cam.columns if 'tail_angle' in c]
    
    # *** HERE IS THE SWITCH FOR MEGABOUTS ***
    bout_results = analyze_behavior(df_cam[tail_cols].values, fs=fps, use_megabouts=MEGABOUTS_FLAG)
    
    # --- 4. Merge Data ---
    cam_indexed = df_cam.set_index('frame_number')
    stim_subset = df_stim[['cam_frame', 'grating_orientation', 'grating_speed']]
    stim_subset = stim_subset.drop_duplicates(subset='cam_frame').set_index('cam_frame')
    
    merged = cam_indexed.join(stim_subset, how='left')
    merged['grating_orientation'] = merged['grating_orientation'].fillna(method='ffill')
    merged['grating_speed'] = merged['grating_speed'].fillna(method='ffill')
    
    # Append Bout Data (Signals)
    for i in range(bout_results['cumul_tail'].shape[1]):
        merged[f'cumul_tail_angle.{i}'] = bout_results['cumul_tail'][:, i]
        merged[f'smooth_cumul_tail_angle.{i}'] = bout_results['smooth_tail'][:, i]

    # Append Bout Data (Indices)
    bout_array = np.zeros(len(merged))
    frame_lookup = df_cam['frame_number'].values
    
    # Map array indices back to frame numbers
    merged['id_bout_start_ind'] = np.nan
    merged['id_bout_end_ind'] = np.nan
    
    for k, (s, e) in enumerate(zip(bout_results['bout_starts'], bout_results['bout_ends'])):
        if s < len(frame_lookup) and e < len(frame_lookup):
            f_s, f_e = frame_lookup[s], frame_lookup[e]
            
            # Map start/end specific frames
            if f_s in merged.index: merged.at[f_s, 'id_bout_start_ind'] = k
            if f_e in merged.index: merged.at[f_e, 'id_bout_end_ind'] = k
            
            # Map active duration
            if e < len(bout_array):
                bout_array[s:e] = k + 1

    merged['tail_active'] = bout_array
    
    # --- 5. Cleanup & Save ---
    drop_cols = ['fish_blob_val', 'max_val', 'x_body_vect', 'y_body_vect'] + [f'tail_value.{i}' for i in range(10)]
    merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)
    merged.to_pickle(final_merged, compression='infer')
    
    print(f"  -> Saved: {final_merged}")
    return True

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    root_path = 'F:/OMR_Ontogeny_VOL'
    
    # Generate database from file structure
    df_experiments = generate_database(root_path)
    
    if not df_experiments.empty:
        # Filter for missing logs
        to_process = df_experiments[df_experiments['saving_flag'] == False]
        print(f"Total Recordings: {len(df_experiments)}")
        print(f"Missing Logs: {len(to_process)}")
        
        # Run pipeline
        for idx, row in to_process.iterrows():
            try:
                process_recording(row)
            except Exception as e:
                print(f"  [CRITICAL FAIL] {row['fish_filename']}: {e}")