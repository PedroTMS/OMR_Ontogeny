import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
import math
import random
import warnings
from scipy import signal
from scipy.ndimage import maximum_filter1d

# Suppress runtime warnings for cleaner output
warnings.filterwarnings('ignore')

# ==============================================================================
# SECTION 1: HELPER ALGORITHMS
# Algorithms for geometry (distance to border) and behavior (bout detection)
# ==============================================================================

def make_circle(points):
    """
    Computes the Smallest Enclosing Circle for a set of 2D points.
    
    Args:
        points (list of tuples): List of (x, y) coordinates.
        
    Returns:
        tuple: (center_x, center_y, radius) of the enclosing circle.
    """
    # Randomized incremental algorithm (Welzl's algorithm)
    # Expected time complexity: O(N)
    shuffled = [(float(x), float(y)) for (x, y) in points]
    random.shuffle(shuffled)
    
    c = None
    for (i, p) in enumerate(shuffled):
        if c is None or not is_in_circle(c, p):
            c = _make_circle_one_point(shuffled[: i + 1], p)
    return c

def _make_circle_one_point(points, p):
    c = (p[0], p[1], 0.0)
    for (i, q) in enumerate(points):
        if not is_in_circle(c, q):
            if c[2] == 0.0:
                c = make_diameter(p, q)
            else:
                c = _make_circle_two_points(points[: i + 1], p, q)
    return c

def _make_circle_two_points(points, p, q):
    circ = make_diameter(p, q)
    left = None
    right = None
    px, py = p
    qx, qy = q
    
    # Check all points to ensure they fit in the circle defined by p, q
    for r in points:
        if is_in_circle(circ, r):
            continue
        
        # Determine if we need to modify the boundary
        cross = _cross_product(px, py, qx, qy, r[0], r[1])
        c = make_circumcircle(p, q, r)
        
        if c is None: continue
        elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
            left = c
        elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
            right = c
            
    if left is None and right is None: return circ
    elif left is None: return right
    elif right is None: return left
    else: return left if (left[2] <= right[2]) else right

def make_circumcircle(p0, p1, p2):
    """Calculates a circle that passes through three points."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0: return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    r = math.hypot(ux - ax, uy - ay)
    return (ux, uy, r)

def make_diameter(p0, p1):
    """Calculates a circle defined by two points as the diameter."""
    cx = (p0[0] + p1[0]) / 2.0
    cy = (p0[1] + p1[1]) / 2.0
    r = math.hypot(cx - p0[0], cy - p0[1])
    return (cx, cy, r)

def is_in_circle(c, p):
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * (1 + 1e-14)

def _cross_product(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

def compute_distance2border(x, y, circle):
    """Calculates the Euclidean distance from a point (x,y) to the edge of the circle."""
    dist2center = np.sqrt((x - circle[0])**2 + (y - circle[1])**2)
    return circle[2] - dist2center

def find_outlier_trajectory(df, max_angular_speed, max_speed, smoothing_win=10):
    """
    Identifies tracking errors where the fish appears to move impossibly fast.
    
    Returns:
        np.array: Boolean array (as floats) indicating invalid frames.
    """
    # Calculate speeds
    angular_speed = 180 / np.pi * np.abs(np.diff(np.unwrap(df['body_angle'].values), prepend=0))
    linear_speed = np.sqrt(np.diff(df['x_pos'].values, prepend=0)**2 + np.diff(df['y_pos'].values, prepend=0)**2)
    
    # Flag invalid frames
    invalid_indices = np.union1d(
        np.where(angular_speed > max_angular_speed)[0],
        np.where(linear_speed > max_speed)[0]
    )
    
    # Smooth the error flags (if one frame is bad, neighbors likely are too)
    fail_mask = np.zeros(len(df))
    fail_mask[invalid_indices] = 1
    smoothed_fail = signal.lfilter(np.ones(smoothing_win), 1, fail_mask) > 0
    
    return smoothed_fail.astype(float)

def detect_tail_bouts(tail_angle_array, fs=700):
    """
    Detects swimming bouts based on tail curvature energy.
    
    Args:
        tail_angle_array (np.array): (n_frames x n_segments) array of tail angles.
        fs (int): Sampling frequency (default 700Hz).
        
    Returns:
        dict: Processed tail metrics and bout indices.
    """
    # 1. Preprocessing: Handle NaNs and Cumulative Sum
    tail_data = tail_angle_array.copy()
    tail_data[np.isnan(tail_data)] = 0
    cumul_tail = np.cumsum(tail_data, axis=1)
    
    # 2. Smoothing (Savgol filter)
    # Smooths across segments to reduce tracking jitter
    smooth_tail = np.zeros_like(cumul_tail)
    for i in range(cumul_tail.shape[1]):
        smooth_tail[:, i] = signal.savgol_filter(cumul_tail[:, i], window_length=11, polyorder=2)
        
    # 3. Calculate "Motion Energy" (curvature change)
    # Diff across time to get velocity-like measure
    diff_tail = np.diff(smooth_tail, axis=0, prepend=0)
    
    # Convolve over tail length (sum activity across all segments)
    # We take absolute diffs to capture magnitude of bend
    motion_signal = np.sum(np.abs(diff_tail), axis=1)
    
    # 4. Filter Signal (Boxcar) to smooth out individual tail beats
    # This creates a "bout envelope"
    boxcar_size = 10
    motion_envelope = signal.convolve(motion_signal, np.ones(boxcar_size)/boxcar_size, mode='same')
    
    # 5. Thresholding for Bouts
    # Constants derived from original MATLAB script logic
    bout_thresh = 0.1
    min_length = 40  # frames
    min_amp = 0.25
    
    # Enhance envelope (Max filter - Min filter strategy)
    # Finds local peaks vs local baselines
    max_filt = maximum_filter1d(motion_envelope, size=20)
    min_filt = -maximum_filter1d(-motion_envelope, size=400)
    enhanced_signal = max_filt - min_filt
    
    # Binary threshold
    is_bout = (enhanced_signal > bout_thresh).astype(int)
    
    # Find start/end indices
    diff_bout = np.diff(is_bout, prepend=0)
    starts = np.where(diff_bout == 1)[0]
    ends = np.where(diff_bout == -1)[0]
    
    # Cleanup edge cases
    if len(starts) > len(ends): starts = starts[:-1]
    if len(ends) > len(starts): ends = ends[1:]
    
    # 6. Filter by Duration and Amplitude
    valid_bouts = []
    for s, e in zip(starts, ends):
        duration = e - s
        amplitude = np.max(enhanced_signal[s:e])
        if duration > min_length and amplitude > min_amp:
            valid_bouts.append((s, e))
            
    valid_starts = [x[0] for x in valid_bouts]
    valid_ends = [x[1] for x in valid_bouts]
    
    return {
        'cumul_tail': cumul_tail,
        'smooth_tail': smooth_tail,
        'bout_starts': np.array(valid_starts),
        'bout_ends': np.array(valid_ends)
    }

# ==============================================================================
# SECTION 2: METADATA PARSING (Replaces MATLAB CSV/Table generation)
# ==============================================================================

def parse_filename_metadata(filename, folder_path):
    """
    Parses fish metadata purely from the filename string.
    Replicates logic from: Preprocessing_OMR_Ontogeny_VOL.m
    
    Expected Format:
    OMR_Ontogeny_VOL_DATE_STRAIN_TANK_CLUTCH_AGE_PROTOCOL_RES_DIA_RIG.txt
    Example: OMR_Ontogeny_VOL_27_02_19_Giant_Tank2_C10_04dpf_P1_75_66_Atlas000.txt
    """
    meta = {}
    
    # Remove extension
    clean_name = filename.replace('.txt', '').replace('.mat', '')
    parts = clean_name.split('_')
    
    # Basic info
    meta['fish_filename'] = filename
    meta['fish_folder'] = folder_path
    
    # Using negative indexing to handle variable prefix lengths safely
    # Mapping based on MATLAB 'end-N' logic
    try:
        # Rig is at the end (e.g., Atlas000 -> Atlas)
        # MATLAB: rig=filename(id(end)+1:end);
        rig_raw = parts[-1]
        meta['fish_rig'] = ''.join([c for c in rig_raw if not c.isdigit()])
        
        # Diameter (e.g., 66)
        # MATLAB: str2num(filename(id(end-1)+1:id(end)-1));
        meta['fish_circle'] = float(parts[-2])
        
        # Resolution (e.g., 75)
        # MATLAB: str2num(filename(id(end-2)+1:id(end-1)-1));
        meta['fish_resolution'] = float(parts[-3])
        
        # Protocol/TimePeriod (e.g., P1 -> 1)
        # MATLAB: str2num(filename(id(end-3)+2:id(end-2)-1));
        meta['fish_timeperiod'] = int(parts[-4].replace('P', ''))
        
        # Age (e.g., 04dpf -> 4)
        # MATLAB: str2num(filename(id(end-4)+1:id(end-3)-4));
        meta['fish_age'] = int(parts[-5].replace('dpf', ''))
        
        # Clutch (e.g., C10 -> 10)
        # MATLAB: str2num(filename(id(end-5)+2:id(end-4)-1));
        meta['fish_clutch'] = int(parts[-6].replace('C', ''))
        
        # Tank (e.g., Tank2 -> 2)
        # MATLAB: str2num(filename(id(end-6)+5:id(end-5)-1));
        meta['fish_tank'] = int(parts[-7].replace('Tank', ''))
        
        # Strain (e.g., Giant)
        # MATLAB: filename(id(end-7)+1:id(end-6)-1);
        meta['fish_strain'] = parts[-8]
        
        # Date (e.g., 27_02_19)
        # MATLAB: filename(id(end-10)+1:id(end-7)-1);
        # Python: join 3 parts preceding the strain
        meta['fish_date'] = "_".join(parts[-11:-8])
        
    except (IndexError, ValueError) as e:
        print(f"  [Warning] Could not parse metadata for {filename}. Using defaults. Error: {e}")
        # Defaults if parsing fails
        meta.update({'fish_resolution': 78.0, 'fish_circle': 66.0, 'fish_strain': 'Unknown'})
        
    return meta

def generate_database(root_folder):
    """
    Crawls the directory to find all fish and statuses.
    Returns a Pandas DataFrame instead of saving a CSV/MAT file.
    """
    records = []
    
    print(f"Scanning {root_folder} for fish experiments...")
    
    for root, dirs, files in os.walk(root_folder):
        # Identify fish by the main text log (ends in 000.txt usually)
        txt_files = [f for f in files if f.endswith('000.txt') and f.startswith('OMR_Ontogeny_VOL_')]
        
        for f in txt_files:
            # 1. Parse Metadata
            entry = parse_filename_metadata(f, root)
            
            # 2. Find Stimulus Log
            # Logic: Look for 'stimlog' in the same folder
            stim_files = [sf for sf in os.listdir(root) if 'stimlog' in sf and sf.endswith('.mat')]
            if stim_files:
                entry['fish_stimlog_filename'] = stim_files[0].replace('.mat', '')
            else:
                entry['fish_stimlog_filename'] = None
                
            # 3. Check Saving Status
            # We look for the final output file: '_MergedLog.pickle'
            base_name = f.replace('.txt', '')
            merged_path = os.path.join(root, base_name + '_MergedLog.pickle')
            
            entry['saving_flag'] = os.path.exists(merged_path)
            
            records.append(entry)
            
    df = pd.DataFrame(records)
    print(f"Database generated. Found {len(df)} recordings.")
    return df

# ==============================================================================
# SECTION 3: LOG CONVERSION & MERGING (Pipeline Logic)
# ==============================================================================

def process_recording(row):
    """
    Executes Step 1 (Convert) -> Step 2 (Bouts) -> Step 3 (Merge) for a single row.
    """
    folder = row['fish_folder']
    base_name = row['fish_filename'].replace('.txt', '')
    
    # ---------------------------------------------------------
    # A. PATH DEFINITIONS
    # ---------------------------------------------------------
    raw_cam_mat = os.path.join(folder, base_name + '.mat')
    pkl_cam = os.path.join(folder, base_name + '.pickle')
    
    stim_name = row['fish_stimlog_filename']
    if not stim_name:
        print(f"  [Skipping] No stimulus log found for {base_name}")
        return False
        
    raw_stim_mat = os.path.join(folder, stim_name + '.mat')
    pkl_stim = os.path.join(folder, stim_name + '.pickle')
    
    final_merged = os.path.join(folder, base_name + '_MergedLog.pickle')

    print(f"Processing {base_name}...")

    # ---------------------------------------------------------
    # B. CONVERT STIMULUS LOG (.mat -> .pickle)
    # ---------------------------------------------------------
    if not os.path.exists(pkl_stim):
        try:
            # MATLAB structs are nested. squeeze_me=True helps flatten them.
            mat = sio.loadmat(raw_stim_mat, struct_as_record=False, squeeze_me=True)
            stim_data = mat['StimLog']
            
            # The Matlab struct has .Values (cells) and .Names.
            # We must iterate and build a dict.
            data_dict = {}
            # Assuming aligned length
            length = stim_data.Length
            
            for i in range(len(stim_data.Values)):
                name = stim_data.Names[i]
                vals = stim_data.Values[i].values
                # Handle varying lengths by right-aligning to the total log length
                # (MATLAB code logic: index = LogLength - numId : LogLength)
                pad_len = length - len(vals)
                padded_vals = np.pad(vals, (pad_len, 0), mode='constant', constant_values=np.nan)
                data_dict[name] = padded_vals
            
            df_stim = pd.DataFrame(data_dict)
            
            # Standardize Column Names (Mapping from MATLAB to Python convention)
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
            print(f"  [Error] Failed converting StimLog: {e}")
            return False
    else:
        df_stim = pd.read_pickle(pkl_stim)

    # ---------------------------------------------------------
    # C. CONVERT CAMERA LOG (.mat -> .pickle)
    # ---------------------------------------------------------
    if not os.path.exists(pkl_cam):
        try:
            # Camera logs are usually HDF5 based inside .mat (v7.3)
            with h5py.File(raw_cam_mat, 'r') as f:
                # Transpose to get (N_samples, N_features)
                cam_data = f['a'][()].T 
                
            # Manual Column Mapping based on '1_Convert_MatFiles.ipynb'
            # Note: 0-based indexing in Python vs 1-based in MATLAB comments
            col_names = [
                'frame_number', 'x_pos', 'y_pos', 'x_body_vect', 'y_body_vect', 'body_angle',
                'max_val', 'fish_blob_val', 'mid_eye_x', 'mid_eye_y', 'hour', 'minute',
                'stim_number', 'cum_sum_tail'
            ]
            # Tail values (10 segments)
            col_names += [f'tail_value.{i}' for i in range(10)]
            col_names.append('body_angle_origin')
            # Tail angles (10 segments)
            col_names += [f'tail_angle.{i}' for i in range(10)]
            col_names += ['first_or_not', 'timing', 'lag']
            
            # Create DF (truncate if mismatch in columns vs data width)
            width = cam_data.shape[1]
            if width == len(col_names):
                df_cam = pd.DataFrame(cam_data, columns=col_names)
            else:
                # Fallback if column counts don't align exactly
                df_cam = pd.DataFrame(cam_data)
                df_cam.columns = col_names[:width] 

            df_cam.to_pickle(pkl_cam, compression='infer')
            
        except Exception as e:
            print(f"  [Error] Failed converting CamLog: {e}")
            return False
    else:
        df_cam = pd.read_pickle(pkl_cam)

    # ---------------------------------------------------------
    # D. BEHAVIORAL ANALYSIS (Bouts & Geometry)
    # ---------------------------------------------------------
    
    # 1. Distance to Border
    pix_size = row['fish_resolution']
    x_mm = df_cam['x_pos'] * pix_size
    y_mm = df_cam['y_pos'] * pix_size
    
    # Subsample points to estimate the arena circle
    points = [(x_mm[i], y_mm[i]) for i in range(0, len(x_mm), 100) if not np.isnan(x_mm[i])]
    if len(points) > 5:
        arena_circle = make_circle(points) # Returns (x, y, r)
        dist_mm = compute_distance2border(x_mm, y_mm, arena_circle)
        df_cam['distance2border_mm'] = dist_mm / 1000.0 # Convert back to mm units
    else:
        df_cam['distance2border_mm'] = np.nan

    # 2. Tracking Failures
    # Logic from 'FindOutlierTrajectory'
    # Speed thresholds (deg/frame and px/frame)
    fps = 700.0
    thresh_ang = (40.0 * 1000) / fps # deg/frame
    thresh_lin = (100.0 / (pix_size/1000.0)) / fps # px/frame
    
    fail_mask = find_outlier_trajectory(df_cam, thresh_ang, thresh_lin)
    df_cam['fail_body_tracking'] = fail_mask

    # 3. Detect Tail Bouts
    # Extract tail columns
    tail_cols = [c for c in df_cam.columns if 'tail_angle' in c]
    tail_data = df_cam[tail_cols].values
    
    bout_results = detect_tail_bouts(tail_data, fs=fps)
    
    # ---------------------------------------------------------
    # E. MERGE LOGS
    # ---------------------------------------------------------
    
    # Index by frame number for alignment
    cam_indexed = df_cam.set_index('frame_number')
    
    # Stimulus log might need frame alignment.
    # Usually we use 'cam_frame' in stim log to match 'frame_number' in cam log.
    stim_subset = df_stim[['cam_frame', 'grating_orientation', 'grating_speed', 'stim_iter']]
    # Remove duplicates where stim didn't update but camera did
    stim_subset = stim_subset.drop_duplicates(subset='cam_frame')
    stim_indexed = stim_subset.set_index('cam_frame')
    
    # Join (Left join on camera data)
    merged = cam_indexed.join(stim_indexed, how='left')
    
    # Forward fill stimulus data (stimulus persists until changed)
    merged['grating_orientation'] = merged['grating_orientation'].fillna(method='ffill')
    merged['grating_speed'] = merged['grating_speed'].fillna(method='ffill')
    
    # Add Bout Data
    # 1. Add cumulative tail angles (calculated during bout detection)
    for i in range(bout_results['cumul_tail'].shape[1]):
        merged[f'cumul_tail_angle.{i}'] = bout_results['cumul_tail'][:, i]
        
    # 2. Map bout starts/ends to the dataframe
    # We create a boolean mask or index column
    merged['bout_id'] = 0
    starts = bout_results['bout_starts']
    ends = bout_results['bout_ends']
    
    # Map array indices back to frame numbers (assuming array is contiguous)
    # merged.index is frame_number. We need to access by integer location (iloc) logic 
    # to place the bout markers, then map back to index.
    
    bout_array = np.zeros(len(merged))
    # Valid bouts only
    for k, (s, e) in enumerate(zip(starts, ends)):
        if e < len(bout_array):
            bout_array[s:e] = k + 1 # Bout ID 1-based
            
    merged['bout_id'] = bout_array
    
    # ---------------------------------------------------------
    # F. SAVE
    # ---------------------------------------------------------
    
    # Drop raw columns to save space if needed
    drop_cols = ['fish_blob_val', 'max_val', 'x_body_vect', 'y_body_vect']
    merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)
    
    merged.to_pickle(final_merged, compression='infer')
    print(f"  -> Saved: {final_merged}")
    return True

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # 1. Define Root (User Input)
    data_folder = 'F:/OMR_Ontogeny_VOL'
    
    # 2. Generate the "Table" dynamically
    df_experiments = generate_database(data_folder)
    
    # 3. Filter for unprocessed fish
    # (saving_flag = False means _MergedLog.pickle does not exist)
    to_process = df_experiments[df_experiments['saving_flag'] == False]
    
    print(f"\nTotal Experiments Found: {len(df_experiments)}")
    print(f"Missing Merged Logs: {len(to_process)}")
    
    # 4. Run Pipeline
    for idx, row in to_process.iterrows():
        try:
            process_recording(row)
        except Exception as e:
            print(f"  [CRITICAL FAIL] Could not process {row['fish_filename']}: {e}")