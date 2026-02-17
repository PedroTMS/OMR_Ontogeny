# Main code file to generate analysis Histograms
"""
This code is an analysis pipeline that walks the Ontogeny dataset to
generate behavioral histograms. It extracts bout metrics (duration,
interbout interval) from existing manually processed data and, as an
option, the user can run Megabouts re-segmentation for direct method
comparison. Output includes two pickled DataFrames: one aggregated
across all conditions and another grouped by stimulus speed, mirroring
the original MATLAB histogram data structures.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import medfilt

# --- CONFIGURATION ---
ROOT_PATH = r'F:\OMR_Ontogeny_VOL' # Update specific root path
SAVE_PATH = Path("dataset") # Folder to save output .pkl files
SAVE_FLAG = True # Set to True to save the results
SAVE_NAME_ALL = 'Analysis_All_Histograms_v2.pkl'
SAVE_NAME_BYSPEED = 'Analysis_BySpeed_Histograms_v2.pkl'

# Analysis Parameters (Matching MATLAB 'make_bout_histograms')
FPS = 700.0 # acquisition frame rate
BIN_SIZE = 0.05 # Seconds
MAX_DURATION = 2.0 # Seconds
BIN_EDGES = np.arange(0, MAX_DURATION + BIN_SIZE, BIN_SIZE)
ALLOWED_SPEEDS = [0.0, 3.0, 5.0, 10.0, 15.0, 30.0] # Global fixed speeds

# Megabouts Flags
RUN_MEGABOUTS = True # Set False to only analyze existing manual data

# Default Parameters for Detection
# (Can be overridden per-fish if needed in the recompute_bouts function)
MEGABOUTS_DEFAULTS = {
    'bout_thresh': 0.1, # [IGNORED] Now calculated dynamically per fish
    'min_duration_frames': 40, # ~57ms @ 700fps
    'savgol_window_ms': 15, # Smoothing window (Library Default)
    'tail_speed_filter_ms': 50, # [UPDATED] Lowered to 50ms to be more responsive
}

# For dynamic bout thresholding
MULTIPLIER = 8.0 # value to multiply to the median

# Conditional Import for Megabouts
try:
    from megabouts.segmentation import TailSegmentation
    from megabouts.preprocessing.tail_preprocessing import TailPreprocessing
    from megabouts.config.segmentation_config import TailSegmentationConfig
    from megabouts.config.preprocessing_config import TailPreprocessingConfig
    MEGABOUTS_AVAILABLE = True
except ImportError:
    MEGABOUTS_AVAILABLE = False
    print("Warning: Megabouts library not found. 'Megabouts' columns will be empty.")

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def clean_tail_angles(df_tail): # Function to clean tracking artifacts
    """
    Applies aggressive cleaning to fix tracking artifacts:
    1. Unwraps angles (fixes 180->-180 degree jumps).
    2. Median filters (removes 1-frame 'teleportation' spikes).
    """
    angles = df_tail.values.copy()
    
    # 1. Unwrap Angles (Fix 180->-180 jumps)
    # This ensures a move from 3.14 to -3.14 is treated as a small change
    angles = np.unwrap(angles, axis=0)
    
    # 2. Median Filter (Fix Teleportation Spikes)
    # Kernel=5 removes short glitches while preserving real bouts
    for i in range(angles.shape[1]):
        angles[:, i] = medfilt(angles[:, i], kernel_size=5)
        
    return pd.DataFrame(angles, columns=df_tail.columns)

def parse_filename_info(filename, fish_counter):
    """
    Parses F-Drive filenames and assigns a sequential FishID.
    Uses negative indexing to be robust against prefix variations.
    
    Expected format: ..._Species_Tank_Clutch_Age_Fish_Res_ArenaSize_Rig_MergedLog.pickle
    Example: OMR_Ontogeny_VOL_27_02_19_Giant_Tank2_C10_04dpf_P1_75_66_Atlas000_MergedLog.pickle
    
    Args:
        filename (str): Raw filename.
        fish_counter (int): Sequential number (1, 2, 3...)
        
    Returns:
        dict: Metadata with synthesized 'FishID'.
    """
    try:
        # 1. Clean filename
        clean_name = os.path.splitext(filename)[0]
        if clean_name.endswith('_MergedLog'):
            clean_name = clean_name.replace('_MergedLog', '')
            
        parts = clean_name.split('_')
        
        # 2. Assign Sequential ID (001, 002...)
        # This creates a clean ID consistent with the histogram structure
        fish_id_str = f"{fish_counter:03d}" 
        
        # 3. Extract Metadata using Negative Indexing
        # Based on structure: ..._Giant(8)_Tank2(7)_C10(6)_04dpf(5)_P1(4)_75(3)_66(2)_Atlas000(1)
        
        meta = {
            'FishID': fish_id_str, # Generated ID
            'Species': parts[-8], # 'Giant'
            'Age': int(parts[-5].replace('dpf', '')), # '04dpf' -> 4
            
            # Extract Rig ('Atlas000' -> 'Atlas')
            'Rig': ''.join([c for c in parts[-1] if not c.isdigit()]) 
        }
        return meta
    except Exception as e:
        # print(f"Parsing Error for {filename}: {e}") 
        return None


def get_bout_metrics(starts, ends, fps, stim_speed_array):
    """
    Calculates physical metrics from start/end frames and aligns stimulus speed.
    
    Args:
        starts (np.array): Array of start frames.
        ends (np.array): Array of end frames.
        fps (float): Frames per second.
        stim_speed_array (np.array): Array of grating speeds per frame.
        
    Returns:
        tuple: (durations, ibis, bout_speeds) all as numpy arrays.
    """
    if len(starts) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 1. Durations (Seconds)
    durations = (ends - starts) / fps
    
    # 2. Interbout Intervals (Seconds)
    # IBI is time from End of Bout N to Start of Bout N+1
    if len(starts) > 1:
        ibis = (starts[1:] - ends[:-1]) / fps
    else:
        ibis = np.array([])
        
    # 3. Stimulus Speed Association
    # Assign speed based on the frame where the bout STARTS.
    # Clip indices to be safe, though valid tracking implies valid frames.
    safe_indices = np.clip(starts.astype(int), 0, len(stim_speed_array) - 1)
    bout_speeds = stim_speed_array[safe_indices] # stimulus speed during each bout
    
    return durations, ibis, bout_speeds


def calculate_histogram_cols(durations, ibis, bin_edges):
    """
    Computes Counts and Probability Density histograms for storing in the DataFrame.
    
    Args:
        durations (np.array): List of bout durations.
        ibis (np.array): List of interbout intervals.
        bin_edges (np.array): The edges for the histogram bins.
        
    Returns:
        dict: Dictionary containing the 4 histogram arrays (Bout/IBI x Counts/Prob).
    """
    def calc_hist(data):
        """
        Helper to calculate raw counts and probability density for a dataset.

        Args:
            data (np.array): Input data array (durations or IBIs).

        Returns:
            tuple: (counts, prob) arrays.
        """
        if len(data) == 0:
            return np.zeros(len(bin_edges)-1), np.zeros(len(bin_edges)-1)
        
        counts, _ = np.histogram(data, bins=bin_edges)
        
        # Normalize (Probability Density)
        total = np.sum(counts)
        if total > 0:
            prob = counts / total
        else:
            prob = np.zeros_like(counts, dtype=float)
            
        return counts, prob

    b_counts, b_prob = calc_hist(durations)
    i_counts, i_prob = calc_hist(ibis)
    
    return {
        'Bout_Counts': b_counts,
        'Bout_Prob': b_prob,
        'IBI_Counts': i_counts,
        'IBI_Prob': i_prob
    }

# ==============================================================================
# 2. MAIN PIPELINE
# ==============================================================================

def main():
    """
    Main execution loop.
    1. Determines allowed speeds from data.
    2. Walks directory structure to find fish.
    3. Extracts metrics using both Manual (existing) and Megabouts (new) methods.
    4. Aggregates data into two DataFrames matching MATLAB structures.
    5. Saves results.
    """
    
    # --- A. INITIALIZATION ---
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    fish_records = [] # Storage for intermediate results
    fish_counter = 1  # Initialize sequential fish counter to add a FishID to filename format
    
    print(f"\nProcessing Fish in {ROOT_PATH}...")
    
    # --- B. PROCESSING LOOP ---
    for root, dirs, files in os.walk(ROOT_PATH):
        merged_files = [f for f in files if f.endswith('_MergedLog.pickle')]
        
        for f in merged_files:
            print(f"  > Found file: {f}")
            file_path = os.path.join(root, f)
            
            # 1. Parse Metadata with Counter
            meta = parse_filename_info(f, fish_counter)
            
            if meta is None:
                print(f"  [!] SKIPPING: Could not parse metadata for {f}")
                continue
                
            # Valid fish found: Increment counter for next fish
            fish_counter += 1
            
            try:
                # 2. Load Data
                print(f"    [1/3] Loading data for Fish {meta['FishID']}...")
                df = pd.read_pickle(file_path)
                
                # Validation
                if 'grating_speed' not in df.columns or 'tail_active' not in df.columns:
                    print(f"Skipping {f}: Missing core columns.")
                    continue
                
                # Get Speed Array (Fill NaNs with 0 to prevent crashes)
                stim_speeds = df['grating_speed'].fillna(0).values
                
                # --- EXTRACT MANUAL DATA ---
                # 'tail_active' is 0 for rest, >0 for bout IDs; tail_Active -> Bout Indices
                tail_active = df['tail_active'].fillna(0).values
                is_active = (tail_active > 0).astype(int)
                diff_active = np.diff(is_active, prepend=0)
                
                man_starts = np.where(diff_active == 1)[0] # extracting the entire array of indices
                man_ends = np.where(diff_active == -1)[0] # extracting the entire array of indices
                
                # Fix edge cases (start without end, end without start)
                if len(man_starts) > len(man_ends):
                    man_starts = man_starts[:-1]
                if len(man_ends) > len(man_starts):
                    man_ends = man_ends[1:]
                
                # Compute Metrics
                man_durs, man_ibis, man_speed_tags = get_bout_metrics(man_starts, man_ends, FPS, stim_speeds)
                
                # --- RUN MEGABOUTS (OPTIONAL) ---
                mega_durs, mega_ibis, mega_speed_tags = [], [], []
                
                if RUN_MEGABOUTS and MEGABOUTS_AVAILABLE:
                    # --- DYNAMIC COLUMN DETECTION ---
                    # Format 1: Underscores (Newer files, e.g., 'tail_angle_0')
                    cols_underscore = [f'tail_angle_{i}' for i in range(10)]
                    # Format 2: Dots (Older files, e.g., 'tail_angle.0')
                    cols_dot = [f'tail_angle.{i}' for i in range(10)]
                    
                    selected_cols = []
                    
                    # Check Format 1
                    if all(c in df.columns for c in cols_underscore):
                        selected_cols = cols_underscore
                    # Check Format 2
                    elif all(c in df.columns for c in cols_dot):
                        selected_cols = cols_dot
                    
                    # Execute if valid columns found
                    if len(selected_cols) == 10:
                        tail_data = df[selected_cols].values
                        
                        # [OPTIMIZATION] Bypass TailTrackingData overhead
                        tail_df_lite = pd.DataFrame(
                            tail_data, 
                            columns=[f"angle_{i}" for i in range(10)]
                        )

                        # --- [UPDATED] ROBUST ADAPTIVE SEGMENTATION LOGIC ---

                        # 1. Clean Data & Interpolate
                        tail_df_lite = tail_df_lite.interpolate(method='linear', limit_direction='both').fillna(0)

                        # 2. Check for Degrees vs Radians and convert if necessary
                        # (This handles the 50x noise difference between species)
                        median_val = tail_df_lite.abs().median().mean()
                        if median_val > 2.0: 
                            tail_df_lite = np.deg2rad(tail_df_lite)

                        # 3. Apply Artifact Cleaning (Unwrap + Median Filter)
                        tail_df_lite = clean_tail_angles(tail_df_lite)
                        
                        # 4. Configure Preprocessing
                        tailprocessing_config = TailPreprocessingConfig(
                            fps=FPS,
                            savgol_window_ms=MEGABOUTS_DEFAULTS['savgol_window_ms'],
                            tail_speed_filter_ms=MEGABOUTS_DEFAULTS['tail_speed_filter_ms']
                        )
                        
                        # 5. Calculate Vigor
                        print("    [2/3] Running Adaptive Megabouts Segmentation...")
                        preprocessor = TailPreprocessing(tailprocessing_config)
                        processed_data = preprocessor.preprocess_tail_df(tail_df_lite)
                        vigor = processed_data.vigor
                        vigor = np.nan_to_num(vigor) # Safety: Remove any remaining NaNs

                        # 6. Dynamic Threshold Calculation
                        # Use 5th percentile as the "True Noise Floor" (resting state)
                        noise_floor = np.percentile(vigor, 5) 
                        if noise_floor < 1.0:
                            noise_floor = 1.0 # Safety floor

                        # MULTIPLIER: 8.0 (Based on our optimization tuning)
                        dynamic_thresh = noise_floor * MULTIPLIER 

                        # 7. Configure Segmentation
                        # Note: Library expects ms, we convert from frames if needed
                        min_dur_ms = (MEGABOUTS_DEFAULTS['min_duration_frames'] * 1000) / FPS
                        
                        tailsegmnt_cfg = TailSegmentationConfig(
                            fps=FPS,
                            min_bout_duration_ms=min_dur_ms,
                            threshold=dynamic_thresh # [UPDATED] Use calculated threshold
                        )
                        
                        # 8. Run Segmentation
                        segmenter = TailSegmentation(tailsegmnt_cfg)
                        results = segmenter.segment(vigor)
                        
                        # 9. Extract Frames
                        mega_starts = np.array(results.onset)
                        mega_ends = np.array(results.offset)
                        
                        # Compute Metrics
                        mega_durs, mega_ibis, mega_speed_tags = get_bout_metrics(mega_starts, mega_ends, FPS, stim_speeds)
                    
                    else:
                        # [DEBUG] Print why it failed
                        print("    [!] SKIPPING MEGABOUTS: Could not find 10 sequential tail columns.")
                        tail_related = [c for c in df.columns if 'tail' in c]
                        print(f"        Available tail columns: {tail_related[:5]} ...")
                
                # --- STORE RAW RECORD ---
                record = {
                    'FishID': meta['FishID'],
                    'Species': meta['Species'],
                    'Age': meta['Age'],
                    'Rig': meta['Rig'],
                    'Filename': f,
                    'Man_Durations': man_durs,
                    'Man_IBIs': man_ibis,
                    'Man_Speeds': man_speed_tags,
                    'Mega_Durations': mega_durs,
                    'Mega_IBIs': mega_ibis,
                    'Mega_Speeds': mega_speed_tags
                }
                fish_records.append(record)
                print(f"    [3/3] Finished Fish {meta['FishID']}. Records: {len(fish_records)}")
                
            except Exception as e:
                print(f"Error processing {f}: {e}")

    if not fish_records:
        print("No valid fish records found. Exiting.")
        return

    # --- C. GENERATE DATASET 1: ALL BOUTS ---
    print(f"\nBuilding 'Analysis_All_Histograms' from {len(fish_records)} fish...")
    
    rows_all = []
    for r in fish_records:
        # Calculate Histograms (All Data)
        hist_manual = calculate_histogram_cols(r['Man_Durations'], r['Man_IBIs'], BIN_EDGES)
        hist_megabouts = calculate_histogram_cols(r['Mega_Durations'], r['Mega_IBIs'], BIN_EDGES)
        
        row = {
            'FishID': r['FishID'],
            'Species': r['Species'],
            'Age': r['Age'],
            'Rig': r['Rig'],
            
            'Manual_Bout_Counts': hist_manual['Bout_Counts'],
            'Manual_Bout_Prob': hist_manual['Bout_Prob'],
            'Manual_IBI_Counts': hist_manual['IBI_Counts'],
            'Manual_IBI_Prob': hist_manual['IBI_Prob'],
            
            'Megabouts_Bout_Counts': hist_megabouts['Bout_Counts'],
            'Megabouts_Bout_Prob': hist_megabouts['Bout_Prob'],
            'Megabouts_IBI_Counts': hist_megabouts['IBI_Counts'],
            'Megabouts_IBI_Prob': hist_megabouts['IBI_Prob']
        }
        rows_all.append(row)
        
    df_all = pd.DataFrame(rows_all)

    # --- D. GENERATE DATASET 2: BY SPEED ---
    print("Building 'Analysis_BySpeed_Histograms'...")
    
    rows_speed = []
    for r in fish_records:
        for speed in ALLOWED_SPEEDS: # [CHANGED] Use global constant instead of 'allowed_speeds'
            tol = 1e-6 # Floating point tolerance
            
            # Filter Manual Data
            idx_manual = np.abs(r['Man_Speeds'] - speed) < tol # of the X bouts, which ones started when the stimulus speed was X
            man_dur_s = r['Man_Durations'][idx_manual]
            
            # IBI Alignment (IBIs are 1 shorter than Bouts)
            if len(r['Man_IBIs']) > 0 and len(r['Man_Speeds']) > 1:
                ibi_speeds = r['Man_Speeds'][:-1]
                idx_manual_ibi = np.abs(ibi_speeds - speed) < tol # of the X ibi, which ones ocurred when the stimulus speed was X
                man_ibi_s = r['Man_IBIs'][idx_manual_ibi]
            else:
                man_ibi_s = np.array([])

            # Filter Megabouts Data
            idx_megabouts = np.abs(r['Mega_Speeds'] - speed) < tol
            mega_dur_s = r['Mega_Durations'][idx_megabouts]
            
            if len(r['Mega_IBIs']) > 0 and len(r['Mega_Speeds']) > 1:
                ibi_speeds_mega = r['Mega_Speeds'][:-1]
                idx_mega_ibi = np.abs(ibi_speeds_mega - speed) < tol
                mega_ibi_s = r['Mega_IBIs'][idx_mega_ibi]
            else:
                mega_ibi_s = np.array([])

            # Compute Histograms (Subset)
            hist_manual_speed = calculate_histogram_cols(man_dur_s, man_ibi_s, BIN_EDGES)
            hist_megabouts_speed = calculate_histogram_cols(mega_dur_s, mega_ibi_s, BIN_EDGES)
            
            row_s = {
                'FishID': r['FishID'],
                'Species': r['Species'],
                'Age': r['Age'],
                'Rig': r['Rig'],
                'Speed': speed,
                
                'Manual_Bout_Counts': hist_manual_speed['Bout_Counts'],
                'Manual_Bout_Prob':   hist_manual_speed['Bout_Prob'],
                'Manual_IBI_Counts':  hist_manual_speed['IBI_Counts'],
                'Manual_IBI_Prob':    hist_manual_speed['IBI_Prob'],
                
                'Megabouts_Bout_Counts': hist_megabouts_speed['Bout_Counts'],
                'Megabouts_Bout_Prob':   hist_megabouts_speed['Bout_Prob'],
                'Megabouts_IBI_Counts':  hist_megabouts_speed['IBI_Counts'],
                'Megabouts_IBI_Prob':    hist_megabouts_speed['IBI_Prob']
            }
            rows_speed.append(row_s)

    df_byspeed = pd.DataFrame(rows_speed)

    # --- E. SAVE RESULTS ---
    if SAVE_FLAG:
        f_all = os.path.join(SAVE_PATH, SAVE_NAME_ALL)
        f_speed = os.path.join(SAVE_PATH, SAVE_NAME_BYSPEED)
        
        print(f"Saving {f_all}...")
        df_all.to_pickle(f_all)
        
        print(f"Saving {f_speed}...")
        df_byspeed.to_pickle(f_speed)
        
        print("Analysis Complete.")

if __name__ == "__main__":
    main()