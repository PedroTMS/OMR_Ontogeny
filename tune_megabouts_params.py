# Simple code to fine tune MEGABOUTS parameters
"""
This code allows to test different MEGABOUTS parameters, to later
fit into the main analysis pipeline. It picks one fish, calculates
the Vigor, plots it so that one can visually see where the threshold
should be. It will compare the "Manual" bouts detection (Ground Truth)
against different Megabouts thresholds.
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

try:
    from megabouts.preprocessing.tail_preprocessing import TailPreprocessing
    from megabouts.config.preprocessing_config import TailPreprocessingConfig
    from megabouts.segmentation import TailSegmentation
    from megabouts.config.segmentation_config import TailSegmentationConfig
except ImportError:
    print("Error: Megabouts library not installed.")
    exit()

# --- CONFIGURATION ---
ROOT_PATH = r'F:\OMR_Ontogeny_VOL' 
FPS = 700.0

# --- TUNING PARAMETERS ---
# INCREASED from 3.0 to 7.0 based on Giant analysis
# Goal: Push threshold (Red Line) above the Median Vigor (Black Line's center)
VIGOR_MULTIPLIER = 7.0  
MEDIAN_FILTER_SIZE = 5  

def load_random_fish():
    """Finds a random _MergedLog.pickle file to analyze."""
    files = []
    print(f"Scanning {ROOT_PATH}...")
    for root, dirs, filenames in os.walk(ROOT_PATH):
        for f in filenames:
            if f.endswith('_MergedLog.pickle'):
                files.append(os.path.join(root, f))
    
    if not files:
        raise FileNotFoundError("No files found!")
    
    choice = np.random.choice(files)
    print(f"Analyzing: {os.path.basename(choice)}")
    return pd.read_pickle(choice)

def clean_tail_angles(df_tail):
    """
    Applies aggressive cleaning to fix tracking artifacts.
    """
    angles = df_tail.values.copy()
    angles = np.unwrap(angles, axis=0) # Fix 180->-180 jumps
    
    print(f"  > Applying Median Filter (Kernel={MEDIAN_FILTER_SIZE})...")
    for i in range(angles.shape[1]):
        angles[:, i] = medfilt(angles[:, i], kernel_size=MEDIAN_FILTER_SIZE)
        
    return pd.DataFrame(angles, columns=df_tail.columns)

def analyze_vigor_and_tune():
    try:
        df = load_random_fish()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # 1. Extract Tail Data
    cols = [f'tail_angle_{i}' for i in range(10)]
    if not all(c in df.columns for c in cols):
        cols = [f'tail_angle.{i}' for i in range(10)]
    
    if not cols:
        print("Error: No tail angle columns found.")
        return

    tail_df = df[cols].copy()
    tail_df.columns = [f"angle_{i}" for i in range(10)]
    
    for c in tail_df.columns:
        tail_df[c] = pd.to_numeric(tail_df[c], errors='coerce')
    
    tail_df = tail_df.interpolate(method='linear', limit_direction='both').fillna(0)
    
    median_val = tail_df.abs().median().mean()
    if median_val > 2.0: 
        print(f"  > Detected DEGREES (Median={median_val:.1f}). Converting to Radians...")
        tail_df = np.deg2rad(tail_df)
    
    # --- APPLY CLEANING ---
    tail_df_clean = clean_tail_angles(tail_df)

    # 2. Calculate Vigor
    print("Calculating Vigor...")
    proc_cfg = TailPreprocessingConfig(fps=FPS, savgol_window_ms=15, tail_speed_filter_ms=50)
    preprocessor = TailPreprocessing(proc_cfg)
    processed = preprocessor.preprocess_tail_df(tail_df_clean)
    vigor = processed.vigor
    vigor = np.nan_to_num(vigor)

    # 3. Dynamic Threshold Calculation
    # Using 5th percentile as the "True Noise Floor" (resting state)
    noise_floor = np.percentile(vigor, 5) 
    if noise_floor < 1.0:
        noise_floor = 1.0 
    
    dynamic_thresh = noise_floor * VIGOR_MULTIPLIER
    
    print(f"Vigor Stats -> Median: {np.median(vigor):.2f}, 5th% (Noise): {noise_floor:.2f}")
    print(f"Dynamic Threshold: {dynamic_thresh:.2f} (Multiplier: {VIGOR_MULTIPLIER}x)")
    
    if dynamic_thresh < np.median(vigor):
        print("  WARNING: Threshold is below Median! Likely to over-detect bouts.")

    # 4. Segment
    seg_cfg = TailSegmentationConfig(fps=FPS, min_bout_duration_ms=20, threshold=dynamic_thresh)
    segmenter = TailSegmentation(seg_cfg)
    res = segmenter.segment(vigor)
    
    mega_active = np.zeros_like(vigor)
    for start, end in zip(res.onset, res.offset):
        mega_active[start:end] = 1

    # 5. Manual Truth
    if 'tail_active' in df.columns:
        manual_active = (df['tail_active'].fillna(0).values > 0).astype(int)
    else:
        manual_active = np.zeros_like(vigor)

    # 6. Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    active_idxs = np.where(manual_active)[0]
    if len(active_idxs) == 0:
        active_idxs = np.where(mega_active)[0]
         
    start = active_idxs[0] - 500 if len(active_idxs) > 0 else 1000
    if start < 0:
        start = 0
    end = start + int(5 * FPS) # 5 seconds
    
    if end > len(vigor):
        end = len(vigor)
        start = max(0, end - int(5 * FPS))

    t = np.arange(start, end) / FPS
    
    # Plot Vigor
    ax.plot(t, vigor[start:end], color='black', alpha=0.6, label='Vigor (Cleaned)', linewidth=1)
    
    # Plot Threshold
    ax.axhline(dynamic_thresh, color='red', linestyle='--', label=f'Thresh ({dynamic_thresh:.1f})')
    
    # Plot Manual Truth
    max_val = np.max(vigor[start:end]) if len(vigor[start:end]) > 0 else 1
    ax.fill_between(t, 0, max_val, where=manual_active[start:end]==1, 
                    color='green', alpha=0.3, label='Manual Truth')
    
    # Plot Mega Result
    ax.plot(t, mega_active[start:end] * (max_val * 0.9), color='orange', linewidth=2, label='Megabouts')

    ax.set_title(f"Segmentation Check: {os.path.basename(df.attrs.get('filename', 'File'))}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vigor (a.u.)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_vigor_and_tune()
