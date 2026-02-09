# Input / Output utility functions file
"""
Ontogeny OMR (Optomotor Response) script for I/O utility functions

In this script you can find multiple utility functions defined to 
facilitate reading and parsing files, handling file/directory paths
and extracting structured data from raw I/O for the Ontogeny OMR
experimental data.
"""

import os
import pandas as pd
import scipy.io as sio
import h5py
import numpy as np
import config  # Import configuration for root path and flags

def get_camera_column_names():
    """
    Returns the standard list of column names for the camera log.
    
    Naming Convention:
      - Uses underscore indexing for both values and angles.
      - Examples: 'tail_value_0', 'tail_angle_0'
      - This ensures consistency with the cleanup logic in processing.py.
    
    Returns:
        list: List of 38 column strings.
    """
    # 1. Standard Metrics (14 cols)
    base_cols = [
        'frame_number', 'x_pos', 'y_pos', 'x_body_vect', 'y_body_vect', 'body_angle',
        'max_val', 'fish_blob_val', 'mid_eye_x', 'mid_eye_y', 'hour', 'minute',
        'stim_number', 'cum_sum_tail'
    ]
    
    # 2. Tail Values (10 cols): tail_value_0 ... tail_value_9
    tail_values = [f'tail_value_{i}' for i in range(10)]
    
    # 3. Reference Angle (1 col)
    angle_origin = ['body_angle_origin']
    
    # 4. Tail Angles (10 cols): tail_angle_0 ... tail_angle_9
    tail_angles = [f'tail_angle_{i}' for i in range(10)]
    
    # 5. Technical Flags (3 cols)
    flags = ['first_or_not', 'timing', 'lag']
    
    return base_cols + tail_values + angle_origin + tail_angles + flags

def parse_filename_metadata(filename, folder_path):
    """
    Extracts experimental parameters from the filename string.
    Strictly parses structure; returns None if any field is missing/malformed.
    
    Args:
        filename (str): Name of the file.
        folder_path (str): Directory path.
        
    Returns:
        dict: Extracted metadata or None if invalid.
    """
    clean_name = filename.replace('.txt', '').replace('.mat', '')
    parts = clean_name.split('_')
    
    try:
        # Strict parsing - no defaults allowed.
        # We use negative indexing to be robust against prefix variations.
        rig_raw = parts[-1]
        
        meta = {
            'fish_filename': filename,
            'fish_folder': folder_path,
            'fish_rig': ''.join([c for c in rig_raw if not c.isdigit()]),
            'fish_circle': float(parts[-2]),
            'fish_resolution': float(parts[-3]),
            'fish_timeperiod': int(parts[-4].replace('P', '')),
            'fish_age': int(parts[-5].replace('dpf', '')),
            'fish_clutch': int(parts[-6].replace('C', '')),
            'fish_tank': int(parts[-7].replace('Tank', '')),
            'fish_strain': parts[-8],
            'fish_date': "_".join(parts[-11:-8])
        }
        return meta
        
    except (IndexError, ValueError):
        # If any index is missing or any conversion fails, discard this file
        return None

def generate_database(root_folder):
    """
    Scans directory tree to build a table of all fish experiments.
    Discards files with invalid naming conventions and reports them.
    
    Args:
        root_folder (str): Root directory to scan.
        
    Returns:
        pd.DataFrame: Table of experiments and their status.
    """
    files_to_df = []
    discarded_files = []
    
    print(f"Scanning {root_folder} for fish experiments...")
    
    for root, dirs, files in os.walk(root_folder):
        # Filter for raw camera logs
        txt_files = [f for f in files if f.endswith('000.txt') and f.startswith('OMR_Ontogeny_VOL_')]
        
        for f in txt_files: # builds dictionary with filename info; returns None if invalid
            # Parse Metadata
            entry = parse_filename_metadata(f, root)
            
            if entry is None:
                discarded_files.append(f)
                continue # Skip this file and move to the next
            
            # Find associated Stimulus Log
            stim_files = [sf for sf in os.listdir(root) if 'stimlog' in sf and sf.endswith('.mat')]
            entry['fish_stimlog_filename'] = stim_files[0].replace('.mat', '') if stim_files else None
            
            # Check if processing is already done
            base_name = f.replace('.txt', '')
            merged_path = os.path.join(root, base_name + '_MergedLog.pickle')
            entry['saving_flag'] = os.path.exists(merged_path)
            
            files_to_df.append(entry)
    
    # Report discarded files to the user
    if discarded_files:
        print("\n" + "="*50)
        print(f"WARNING: Discarded {len(discarded_files)} files due to invalid naming/typos:")
        for bad_file in discarded_files:
            print(f"  [X] {bad_file}")
        print("="*50 + "\n")
        
    return pd.DataFrame(files_to_df)

def load_stim_log(file_path):
    """
    Loads and parses the Stimulus MAT file into a DataFrame.
    
    Args:
        file_path (str): Full path to the .mat file.
        
    Returns:
        pd.DataFrame: Parsed stimulus data.
    """
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    stim_data = mat['StimLog']
    data_dict = {}
    length = stim_data.Length
    
    for i in range(len(stim_data.Values)):
        name = stim_data.Names[i]
        vals = stim_data.Values[i].values
        # Pad with NaNs if lengths don't match
        pad_len = length - len(vals)
        padded_vals = np.pad(vals, (pad_len, 0), mode='constant', constant_values=np.nan)
        data_dict[name] = padded_vals
    
    return pd.DataFrame(data_dict)

def load_cam_log(file_path):
    """
    Loads the Camera HDF5/MAT file into a DataFrame.
    
    Args:
        file_path (str): Full path to the .mat file.
        
    Returns:
        pd.DataFrame: Raw camera data with generic column names (renaming happens later).
    """
    with h5py.File(file_path, 'r') as f:
        # Load all data, transpose to (Time x Variables)
        cam_data = f['a'][()].T # read everything, transpose and load to memory
    return pd.DataFrame(cam_data)

def addindex2identicalcolumnsname(df):
    """
    Renames duplicate DataFrame columns by appending an underscore counter.
    Example: 'col', 'col' -> 'col_0', 'col_1'
    
    Args:
        df (pd.DataFrame): Input dataframe with potential duplicate columns.
        
    Returns:
        pd.DataFrame: Dataframe with unique column names.
    """
    cols = pd.Series(df.columns)
    for dup in df.columns[df.columns.duplicated()].unique():
        mask = df.columns.get_loc(dup)
        count = mask.sum()
        
        # Use simple underscore indexing for all duplicates.
        # This aligns with the 'tail_value_0' style logic.
        new_names = [f"{dup}_{i}" for i in range(count)]
            
        cols[mask] = new_names
        
    df.columns = cols
    return df