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
import warnings
from scipy import signal

# --- CONFIGURATION ---
ROOT_PATH = r'F:\OMR_Ontogeny_VOL' # Update specific root path
SAVE_PATH = Path("dataset") # Folder to save output .pkl files
SAVE_FLAG = True # Set to True to save the results

# Analysis Parameters (Matching MATLAB 'make_bout_histograms')
FPS = 700.0 # acquisition frame rate
BIN_SIZE = 0.05 # Seconds
MAX_DURATION = 2.0 # Seconds
BIN_EDGES = np.arange(0, MAX_DURATION + BIN_SIZE, BIN_SIZE)

# Megabouts Flags
RUN_MEGABOUTS = True # Set False to only analyze existing manual data

# Default Parameters for Detection
# (Can be overridden per-fish if needed in the recompute_bouts function)
MEGABOUTS_DEFAULTS = {
    'bout_thresh': 0.1, # Sensitivity threshold (Pipeline default)
    'min_duration_frames': 40, # Minimum duration in frames (Pipeline default)
    'savgol_window_ms': 15, # [NEW] Smoothing window (Library Default)
    'tail_speed_filter_ms': 100, # [NEW] Vigor filter size (Library Default)
}

# Conditional Import for Megabouts
try:
    from megabouts.tracking_data import TailTrackingData
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

def get_unique_speeds_from_logs(root_path):
    """
    Scans all _MergedLog.pickle files in the directory structure to 
    dynamically extract the set of unique stimulus speeds used in experiments.
    
    Args:
        root_path (str): The root directory to scan.
        
    Returns:
        list: A sorted list of unique speeds (e.g., [0.0, 3.0, 5.0, ...])
    """
    unique_speeds = set()
    print(f"Scanning {root_path} to determine ALLOWED_SPEEDS...")

    for root, dirs, files in os.walk(root_path):
        merged_files = [f for f in files if f.endswith('_MergedLog.pickle')]
        
        for f in merged_files:
            file_path = os.path.join(root, f)
            try:
                # We interpret the pickle partially or fully. 
                # Reading the full pickle is safer to ensure schema compliance.
                df = pd.read_pickle(file_path)
                
                if 'grating_speed' in df.columns:
                    # Drop NaNs and get unique values
                    raw_speeds = df['grating_speed'].dropna().unique()
                    
                    # Round to 1 decimal place to avoid floating-point mismatch 
                    rounded_speeds = np.round(raw_speeds, 1)
                    
                    unique_speeds.update(rounded_speeds)
            except Exception as e:
                continue

    # Filter out negative artifacts if any, and sort
    final_speeds = sorted([s for s in list(unique_speeds) if s >= 0])
    
    print(f"-> Dynamic speeds detected: {final_speeds}")
    return final_speeds


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
            'FishID': fish_id_str,           # Generated ID
            'Species': parts[-8],            # 'Giant'
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
    # We clip indices to be safe, though valid tracking implies valid frames.
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
        
    # Dynamically determine speeds
    allowed_speeds = get_unique_speeds_from_logs(ROOT_PATH)
    if not allowed_speeds:
        print("Warning: No speeds found. Using default [0, 3, 5, 10, 15, 30].")
        allowed_speeds = [0, 3, 5, 10, 15, 30] # expected speeds from ontogeny_omr protocol
        
    fish_records = [] # Storage for intermediate results
    fish_counter = 1  # Initialize sequential fish counter to add a FishID to filename format
    
    print(f"\nProcessing Fish in {ROOT_PATH}...")
    
    # --- B. PROCESSING LOOP ---
    for root, dirs, files in os.walk(ROOT_PATH):
        merged_files = [f for f in files if f.endswith('_MergedLog.pickle')]
        
        for f in merged_files:
            file_path = os.path.join(root, f)
            
            # 1. Parse Metadata with Counter
            meta = parse_filename_info(f, fish_counter)
            
            if meta is None:
                continue
                
            # Valid fish found: Increment counter for next fish
            fish_counter += 1
            
            try:
                # 2. Load Data
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
                    # Extract Tail Matrix
                    tail_cols = [c for c in df.columns if 'tail_angle' in c]
                    if len(tail_cols) > 0:
                        tail_data = df[tail_cols].values
                        
                        # 1. Configure Preprocessing (NEW Parameters)
                        # Controls smoothing and vigor calculation
                        tailprocessing_config = TailPreprocessingConfig(
                            fps=FPS,
                            savgol_window_ms=MEGABOUTS_DEFAULTS['savgol_window_ms'],
                            tail_speed_filter_ms=MEGABOUTS_DEFAULTS['tail_speed_filter_ms']
                        )
                        
                        # 2. Configure Segmentation
                        # Controls thresholding logic
                        # Note: Library expects ms, we convert from frames if needed
                        min_dur_ms = (MEGABOUTS_DEFAULTS['min_duration_frames'] * 1000) / FPS
                        
                        tailsegmnt_cfg = TailSegmentationConfig(
                            fps=FPS,
                            min_bout_duration_ms=min_dur_ms,
                            threshold=MEGABOUTS_DEFAULTS['bout_thresh']
                        )
                        
                        # 3. Execution Pipeline
                        # A. Preprocessing: Get Tail Vigor
                        # We instantiate the preprocessor with our config
                        preprocessor = TailPreprocessing(tailprocessing_config)
                        
                        # Assuming .process() or .run() takes the raw array and returns object with .tail_vigor
                        # We use tracking data container to be safe
                        tracking_data = TailTrackingData.from_array(tail_data, fps=FPS)
                        processed_data = preprocessor.run(tracking_data) 
                        
                        # B. Segmentation: Get Bouts
                        # Segmenter takes the calculated tail vigor
                        segmenter = TailSegmentation(tailsegmnt_cfg)
                        results = segmenter.segment(processed_data.tail_vigor)
                        
                        # 4. Extract Frames
                        mega_starts = np.array(results.onset)
                        mega_ends = np.array(results.offset)
                        
                        # Compute Metrics
                        mega_durs, mega_ibis, mega_speed_tags = get_bout_metrics(mega_starts, mega_ends, FPS, stim_speeds)
                
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
        for speed in allowed_speeds:
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
        f_all = os.path.join(SAVE_PATH, 'Analysis_All_Histograms.pkl')
        f_speed = os.path.join(SAVE_PATH, 'Analysis_BySpeed_Histograms.pkl')
        
        print(f"Saving {f_all}...")
        df_all.to_pickle(f_all)
        
        print(f"Saving {f_speed}...")
        df_byspeed.to_pickle(f_speed)
        
        print("Analysis Complete.")

if __name__ == "__main__":
    main()