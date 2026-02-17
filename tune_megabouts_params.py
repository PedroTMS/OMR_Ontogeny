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
from pathlib import Path
from megabouts.preprocessing.tail_preprocessing import TailPreprocessing
from megabouts.config.preprocessing_config import TailPreprocessingConfig
from megabouts.segmentation import TailSegmentation
from megabouts.config.segmentation_config import TailSegmentationConfig

# --- CONFIGURATION ---
ROOT_PATH = r'F:\OMR_Ontogeny_VOL'  # Update to your data path
FPS = 700.0

# --- PARAMETERS TO TEST ---
# We will test 3 different thresholds to see which one matches the Manual Data best
TEST_THRESHOLDS = [0.05, 0.1, 0.2] 
TEST_MIN_DURATION_MS = 20 # Lowered from ~57ms to catch shorter events
SAVGOL_WINDOW_MS = 15
TAIL_SPEED_FILTER_MS = 50 # Lowered to make vigor more responsive

def load_random_fish():
    """Finds a random _MergedLog.pickle file to analyze."""
    files = []
    for root, dirs, filenames in os.walk(ROOT_PATH):
        for f in filenames:
            if f.endswith('_MergedLog.pickle'):
                files.append(os.path.join(root, f))
    
    if not files:
        raise FileNotFoundError("No files found!")
    
    # Pick one file (Fixed index or random)
    # Using specific one if known, or random
    choice = np.random.choice(files)
    print(f"Analyzing: {os.path.basename(choice)}")
    return pd.read_pickle(choice)

def analyze_vigor_and_tune():
    df = load_random_fish()
    
    # 1. Extract Tail Data
    # Try underscore then dot format
    cols_underscore = [f'tail_angle_{i}' for i in range(10)]
    cols_dot = [f'tail_angle.{i}' for i in range(10)]
    
    if all(c in df.columns for c in cols_underscore):
        tail_cols = cols_underscore
    elif all(c in df.columns for c in cols_dot):
        tail_cols = cols_dot
    else:
        print("Could not find tail columns.")
        return

    tail_data = df[tail_cols].values
    tail_df_lite = pd.DataFrame(tail_data, columns=[f"angle_{i}" for i in range(10)])

    # 2. Run Preprocessing (Calculate Vigor)
    print("Calculating Vigor...")
    
    # Config
    proc_cfg = TailPreprocessingConfig(
        fps=FPS,
        savgol_window_ms=SAVGOL_WINDOW_MS,
        tail_speed_filter_ms=TAIL_SPEED_FILTER_MS
    )
    
    preprocessor = TailPreprocessing(proc_cfg)
    processed = preprocessor.preprocess_tail_df(tail_df_lite)
    vigor = processed.vigor
    
    print(f"Vigor Stats -> Max: {np.max(vigor):.4f}, Mean: {np.mean(vigor):.4f}, Median: {np.median(vigor):.4f}")

    # 3. Extract Manual Bouts (Ground Truth)
    tail_active = df['tail_active'].fillna(0).values
    manual_is_active = (tail_active > 0).astype(int)

    # 4. Run Megabouts with different thresholds
    results = {}
    for thresh in TEST_THRESHOLDS:
        seg_cfg = TailSegmentationConfig(
            fps=FPS,
            min_bout_duration_ms=TEST_MIN_DURATION_MS,
            threshold=thresh
        )
        segmenter = TailSegmentation(seg_cfg)
        res = segmenter.segment(vigor)
        
        # Convert onset/offset to a binary array for plotting
        mega_active = np.zeros_like(vigor)
        for start, end in zip(res.onset, res.offset):
            mega_active[start:end] = 1
        results[thresh] = mega_active

    # 5. Plot Comparison
    # We plot a random 5-second slice where activity exists
    active_indices = np.where(manual_is_active)[0]
    if len(active_indices) > 0:
        start_frame = active_indices[0] - 500
        if start_frame < 0: start_frame = 0
    else:
        start_frame = 1000
        
    end_frame = start_frame + int(5 * FPS) # 5 seconds window
    time_axis = np.arange(start_frame, end_frame) / FPS
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot Vigor (The signal we are thresholding)
    # We scale it up slightly to match the binary blocks for visibility
    ax.plot(time_axis, vigor[start_frame:end_frame], color='black', alpha=0.6, label='Vigor Signal', linewidth=1)
    
    # Plot Manual Truth (Green blocks)
    ax.fill_between(time_axis, 0, 1, where=manual_is_active[start_frame:end_frame]==1, 
                    color='green', alpha=0.3, label='Manual (Ground Truth)', transform=ax.get_xaxis_transform())

    # Plot Megabouts Estimates (Stepped lines)
    colors = ['red', 'orange', 'blue']
    for i, thresh in enumerate(TEST_THRESHOLDS):
        # Offset the lines slightly so they don't overlap
        y_offset = -0.1 - (i * 0.1)
        data_slice = results[thresh][start_frame:end_frame]
        ax.plot(time_axis, data_slice * 0.5 + y_offset, color=colors[i], label=f'Mega Thresh {thresh}')
        # Draw the threshold line
        ax.axhline(y=thresh, color=colors[i], linestyle=':', alpha=0.5)

    ax.set_title(f"Segmentation Tuning (Window: {start_frame}-{end_frame})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity / Vigor")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_vigor_and_tune()
