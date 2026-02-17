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
try:
    from megabouts.preprocessing.tail_preprocessing import TailPreprocessing
    from megabouts.config.preprocessing_config import TailPreprocessingConfig
    from megabouts.segmentation import TailSegmentation
    from megabouts.config.segmentation_config import TailSegmentationConfig
except ImportError:
    print("Error: Megabouts library not installed or not found in path.")
    exit()

# --- CONFIGURATION ---
ROOT_PATH = r'F:\OMR_Ontogeny_VOL'  # Update to your data path
FPS = 700.0

# --- PARAMETERS TO TEST ---
# Now that we convert to Radians, 0.1 is a reasonable starting point.
# Testing thresholds significantly above the noise floor (Median ~20)
TEST_THRESHOLDS = [20, 40, 80]
TEST_MIN_DURATION_MS = 20 
SAVGOL_WINDOW_MS = 15
TAIL_SPEED_FILTER_MS = 50 

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

def analyze_vigor_and_tune():
    try:
        df = load_random_fish()
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # 1. Extract Tail Data
    cols_underscore = [f'tail_angle_{i}' for i in range(10)]
    cols_dot = [f'tail_angle.{i}' for i in range(10)]
    
    if all(c in df.columns for c in cols_underscore):
        tail_cols = cols_underscore
    elif all(c in df.columns for c in cols_dot):
        tail_cols = cols_dot
    else:
        print("Could not find tail columns.")
        return

    print(f"Found {len(tail_cols)} tail columns. Cleaning data...")
    
    # Create DataFrame and Force Numeric Types
    tail_df_lite = df[tail_cols].copy()
    tail_df_lite.columns = [f"angle_{i}" for i in range(10)] # Rename for Megabouts
    
    for c in tail_df_lite.columns:
        tail_df_lite[c] = pd.to_numeric(tail_df_lite[c], errors='coerce')

    # [STEP 1: REMOVE ARTIFACTS]
    # Filter out impossible angles (e.g. > 360 or < -360) often used as error codes
    # This fixes the "Max: 47179" issue
    mask_artifacts = tail_df_lite.abs() > 360
    if mask_artifacts.sum().sum() > 0:
        print(f"  > Removing {mask_artifacts.sum().sum()} artifact values (>360)...")
        tail_df_lite[mask_artifacts] = np.nan

    # [STEP 2: INTERPOLATE]
    tail_df_lite = tail_df_lite.interpolate(method='linear', limit_direction='both')
    tail_df_lite = tail_df_lite.fillna(0)

    # [STEP 3: CONVERT TO RADIANS]
    # If the median absolute angle is > 1.0, it is almost certainly Degrees.
    # (Radians for tail angles are usually small, e.g., 0.1 - 0.5)
    median_val = tail_df_lite.abs().median().mean()
    if median_val > 1.0:
        print(f"  > Detected DEGREES (Median={median_val:.2f}). Converting to RADIANS...")
        tail_df_lite = np.deg2rad(tail_df_lite)
    else:
        print(f"  > Detected RADIANS (Median={median_val:.2f}). Keeping as is.")

    # 2. Run Preprocessing (Calculate Vigor)
    print("Calculating Vigor...")
    
    proc_cfg = TailPreprocessingConfig(
        fps=FPS,
        savgol_window_ms=SAVGOL_WINDOW_MS,
        tail_speed_filter_ms=TAIL_SPEED_FILTER_MS
    )
    
    try:
        preprocessor = TailPreprocessing(proc_cfg)
        processed = preprocessor.preprocess_tail_df(tail_df_lite)
        vigor = processed.vigor
        
        if np.isnan(vigor).any():
            print("  > Warning: Vigor calculation produced NaNs. Replacing with 0.")
            vigor = np.nan_to_num(vigor)

        print(f"Vigor Stats -> Max: {np.max(vigor):.4f}, Mean: {np.mean(vigor):.4f}, Median: {np.median(vigor):.4f}")
    except Exception as e:
        print(f"Error inside Megabouts preprocessing: {e}")
        return

    # 3. Extract Manual Bouts (Ground Truth)
    if 'tail_active' in df.columns:
        tail_active = df['tail_active'].fillna(0).values
        manual_is_active = (tail_active > 0).astype(int)
    else:
        print("Warning: 'tail_active' not found. Cannot plot Manual Truth.")
        manual_is_active = np.zeros_like(vigor)

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
        
        mega_active = np.zeros_like(vigor)
        for start, end in zip(res.onset, res.offset):
            mega_active[start:end] = 1
        results[thresh] = mega_active

    # 5. Plot Comparison
    active_indices = np.where(manual_is_active)[0]
    if len(active_indices) > 0:
        rand_idx = np.random.choice(active_indices)
        start_frame = max(0, rand_idx - 500)
    else:
        start_frame = 1000
        
    end_frame = start_frame + int(5 * FPS) # 5 seconds window
    
    # Safety Check for indices
    end_frame = min(len(vigor), end_frame)
    if end_frame - start_frame < 100:
        start_frame = 0
        end_frame = min(len(vigor), int(5*FPS))

    time_axis = np.arange(start_frame, end_frame) / FPS
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot Vigor
    ax.plot(time_axis, vigor[start_frame:end_frame], color='black', alpha=0.6, label='Vigor Signal', linewidth=1)
    
    # Plot Manual Truth (Green blocks)
    ax.fill_between(time_axis, 0, 1, where=manual_is_active[start_frame:end_frame]==1, 
                    color='green', alpha=0.3, label='Manual (Ground Truth)', transform=ax.get_xaxis_transform())

    # Plot Megabouts Estimates (Stepped lines)
    colors = ['red', 'orange', 'blue']
    for i, thresh in enumerate(TEST_THRESHOLDS):
        y_offset = -0.1 - (i * 0.1)
        data_slice = results[thresh][start_frame:end_frame]
        ax.plot(time_axis, data_slice * 0.5 + y_offset, color=colors[i], label=f'Mega Thresh {thresh}')
        ax.axhline(y=thresh, color=colors[i], linestyle=':', alpha=0.5)

    ax.set_title(f"Segmentation Tuning (Window: {start_frame}-{end_frame})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vigor")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_vigor_and_tune()
