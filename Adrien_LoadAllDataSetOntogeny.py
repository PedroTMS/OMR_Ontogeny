# library:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import our new loaders and pipeline steps
from ethogram.loaders.loader_ontogeny import ExperimentRegistryOntogeny
from ethogram.pipelines.ontogeny.steps_core.step01_camera_loading import load_cam_log
from ethogram.pipelines.ontogeny.steps_core.step02_stimulus_loading import load_stim_log
from ethogram.pipelines.ontogeny.steps_core.step03_data_merging import merge_logs
from ethogram.pipelines.ontogeny.steps_core.step04_behavior_extraction import run_megabouts, extract_bout_features
from ethogram.pipelines.ontogeny.steps_core.step05_stimulus_processing import compute_stimulus_sequence, extract_stimulus_code
from ethogram.pipelines.ontogeny.steps_glm.step06_glm_preparation import compute_glm_input, downsample_for_glm

# Import utility functions
from ethogram.pipelines.common.utils import find_onset_offset_numpy

# PLOTS TO MAKE:

# CAMLOG CHECK:
# CHECK GEOMETRY: TRAJECTORY VS TAIL VS ANGLE RELATION 

# STIMLOG CHECK:
# CHECK XPOS SHADER VS XPOS CAMLOG
# CHECK FPS

# CHECK MEGABOUTS:
# CHECK SEGMENTATION
# CHECK CLASSIFICATION

# CHECK ETHOGRAM:
# FOR A GIVEN SET OF STIMULI, CHECK THE ETHOGRAM

# CHECK BOUTS TRIGGERED AVERAGE BY STIMULUS


# CHECK ANIMATION: MAKE ANIMATION OF THE FISH AND THE STIMULUS FOR A GIVEN TRIAL

# Make plotting code for the above
 

# Load Table (using loaders)
data_folder = 'F:/'
protocol_name = 'OMR_Ontogeny_VOL'

expe_registry = ExperimentRegistryOntogeny(
    data_folder=data_folder,
    protocol_name=protocol_name,
)
expe_table = expe_registry.table

print(f"Loaded {len(expe_table)} experiments")
print(f"Available strains: {expe_table.strain.unique()}")
print(f"Table columns: {list(expe_table.columns)}")

# Load Fish (using loaders)
# Select first fish from Tu strain
tu_fish = expe_table[expe_table.strain == 'Tu']
fish_id = tu_fish.index[0]  # Get first valid fish ID
fish = expe_registry.get_recording(fish_id)

print(f"Selected fish ID: {fish_id}")
print(f"Fish experiment name: {fish.expe_name}")


#  Load All logs step by step:

# Step 1: Cam log
print("Loading camera log...")
cam_log = load_cam_log(fish)
duration_sec = cam_log.shape[0] / 700
duration_hours = duration_sec / 3600
print(f"Camera log: {cam_log.shape[0]} frames ({duration_sec:.1f}s = {duration_hours:.2f}h at 700Hz)")
print(f"Camera log columns: {list(cam_log.columns)}")
print(f"Camera log time range: frame {cam_log.index.min()} to {cam_log.index.max()}")

# Step 2: Stim log and arduino log
print("\nLoading stimulus and Arduino logs...")
stim_log = load_stim_log(fish)
stim_fps = stim_log.shape[0] / duration_sec if duration_sec > 0 else 0
print(f"Stimulus log: {stim_log.shape[0]} frames (~{stim_fps:.0f}Hz over same time period)")
print(f"Stimulus log columns: {list(stim_log.columns)}")

# Step 3: Image table log
info_log = fish.load_raw_info()
print(f"\nInfo log keys: {list(info_log.keys())}")

# Does it make sense to have stim log and no camlog?
print(f"Camera log time range: frame {cam_log.index.min()} to {cam_log.index.max()}")
print(f"Stimulus log time range: frame {stim_log.index.min()} to {stim_log.index.max()}")

# load MergedLog (using pipeline/steps 1 to 3 )
print("\nMerging all logs...")
merged_log = merge_logs(cam_log, stim_log, fill_limit=300)
print(f"Merged log: {merged_log.shape[0]} frames, {merged_log.shape[1]} columns")
print(f"Merged log columns: {list(merged_log.columns)}")

# Check for missing data in core behavioral/tracking columns (ignore stimulus NaNs which are expected)
core_columns = ['x_pos', 'y_pos', 'body_angle', 'distance2edge', 'angle2edge', 'is_spontaneous']
missing_data = merged_log[core_columns].isnull().sum()
print(f"\nMissing data in core columns:")
any_missing = False
for col, missing in missing_data.items():
    if missing > 0:
        print(f"  {col}: {missing} missing values ({missing/len(merged_log)*100:.1f}%)")
        any_missing = True
if not any_missing:
    print("  ✓ No missing data in core tracking/behavioral columns")


# load Behavior (steps 4, Megabouts)
print("\nRunning megabouts analysis...")
ethogram, bouts, segments, tail, traj = run_megabouts(
    merged_log, 
    exclude_CS=True, 
    peak_prominence=1
)
print(f"✓ Megabouts analysis completed")
print(f"Ethogram shape: {ethogram.df.shape}")
print(f"Number of bouts detected: {len(bouts.df)}")

# Extract bout features
bout_df = extract_bout_features(ethogram, bouts, merged_log)
print(f"Bout features shape: {bout_df.shape}")
print(f"Bout categories: {bout_df.bout_cat.unique()}")