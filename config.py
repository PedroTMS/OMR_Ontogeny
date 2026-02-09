# Configuration
"""
Ontogeny OMR (Optomotor Response) script for seting up basic configurations

This scripts contains global settings used across the pipeline.
Main root path, independent variables and megabouts flag.
"""

import warnings

# --- File System Settings ---
# The main folder containing your experiment data
ROOT_PATH = 'F:/OMR_Ontogeny_VOL'

# --- Experimental Constants ---
FPS = 700.0             # Camera acquisition speed (Frames Per Second)
MICRONS_PER_MM = 1000.0 # Conversion factor
MS_PER_SEC = 1000.0     # Conversion factor

# --- Analysis Thresholds (Outlier Detection) ---
# Maximum physically possible speeds for a zebrafish larva
MAX_ANG_VEL_DEG_PER_MS = 40.0   # Max angular velocity (deg/ms)
MAX_LIN_VEL_MM_PER_S = 100.0    # Max linear velocity (mm/s)
# Bout Detection Thresholds
MIN_BOUT_DURATION = 40  # Minimum duration in frames
MIN_BOUT_AMPLITUDE = 0.25  # Minimum amplitude in accumulated tail signal

# --- Processing Parameters ---
BORDER_SAMPLING_STEP = 100      # Step size for subsampling border points
MIN_CIRCLE_POINTS = 5           # Min points required to define arena circle

# --- Global Flags ---
# Set to True to use the 'megabouts' library for deep-learning detection.
MEGABOUTS_FLAG = False 

# --- Conditional Imports ---
MEGABOUTS_AVAILABLE = False
if MEGABOUTS_FLAG:
    try:
        from megabouts.tracking_data import TailTrackingData
        from megabouts.segmentation import TailSegmentation
        from megabouts.config.segmentation_config import TailSegmentationConfig
        MEGABOUTS_AVAILABLE = True
    except ImportError:
        print("[Warning] MEGABOUTS_FLAG is True but library not found. Falling back to manual detection.")
        MEGABOUTS_AVAILABLE = False

# --- Warning Suppression ---
warnings.filterwarnings('ignore')