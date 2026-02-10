import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# PASTE YOUR PATHS HERE:
OLD_FILE_PATH = r"C:\Users\orger\Desktop\OMR_Ontogeny_VOL_09_03_19_Giant_Tank6_C14_07dpf_P4_76_66_Atlas000_MergedLog.pickle"
NEW_FILE_PATH = r"F:\OMR_Ontogeny_VOL\atlas\Giant_Danio\7dpf\P4\OMR_Ontogeny_VOL_09_03_19_Giant_Tank6_C14_07dpf_P4_76_66_Atlas000_MergedLog.pickle"
# ==========================================

def compare_logs(old_path, new_path):
    print("\n" + "="*60)
    print("MATCHING REPORT: OLD vs NEW LOGS")
    print("="*60)
    print(f"Old File: {os.path.basename(old_path)}")
    print(f"New File: {os.path.basename(new_path)}")
    print("-" * 60)

    # 1. Load Files
    try:
        df_old = pd.read_pickle(old_path)
        df_new = pd.read_pickle(new_path)
    except Exception as e:
        print(f"FATAL ERROR: Could not load files.\n{e}")
        return

    # 2. Shape Comparison
    rows_old, cols_old_len = df_old.shape
    rows_new, cols_new_len = df_new.shape
    
    print("SHAPE CHECK:")
    print(f"  Old: {rows_old} rows, {cols_old_len} cols")
    print(f"  New: {rows_new} rows, {cols_new_len} cols")
    
    if rows_old != rows_new:
        print(f"  [WARNING] Row count mismatch! Diff: {abs(rows_old - rows_new)}")
    else:
        print("  [OK] Row counts match perfectly.")

    # 3. Column Presence Logic
    old_cols = set(df_old.columns)
    new_cols = set(df_new.columns)
    
    missing_in_new = old_cols - new_cols
    extra_in_new = new_cols - old_cols
    
    renamed_count = 0
    truly_missing = []
    
    print("-" * 60)
    print("COLUMN MAPPING ANALYSIS:")
    
    for col in missing_in_new:
        # Check if it was simply renamed (dot to underscore)
        predicted_name = col.replace('.', '_')
        if predicted_name in new_cols:
            renamed_count += 1
            # Optional: print first few examples
            if renamed_count <= 3:
                 print(f"  [RENAME DETECTED] {col}  ->  {predicted_name}")
        else:
            truly_missing.append(col)

    if renamed_count > 0:
        print(f"  ... (Total {renamed_count} columns successfully renamed from dots to underscores)")

    # 4. CRITICAL MISSING DATA CHECK
    print("-" * 60)
    print("MISSING DATA CHECK (The most important part):")
    
    # List of critical vars we specifically care about
    critical_vars = ['stim_number', 'stim_iter', 'x_pos_shader', 'y_pos_shader', 
                     'i_global_time', 'time_cam_shader']
    
    found_issues = False
    for var in critical_vars:
        if var not in df_new.columns:
            print(f"  [CRITICAL MISSING] '{var}' is NOT in the new file!")
            found_issues = True
    
    if truly_missing:
        print("  [WARNING] The following Old columns are completely gone in New file:")
        for col in sorted(truly_missing):
            print(f"    - {col}")
        found_issues = True
    
    if not found_issues:
        print("  [SUCCESS] All critical columns and data fields are present.")

    # 5. Data Integrity Check (Values)
    print("-" * 60)
    print("VALUE INTEGRITY CHECK:")
    
    # Compare a few stable columns to ensure data isn't corrupted
    test_cols = ['x_pos', 'y_pos', 'body_angle', 'hour']
    # Add stim_number if it exists in both
    if 'stim_number' in df_new.columns and 'stim_number' in df_old.columns:
        test_cols.append('stim_number')

    for col in test_cols:
        if col in df_new.columns and col in df_old.columns:
            # Drop NaNs for comparison and ensure numeric
            val_old = pd.to_numeric(df_old[col], errors='coerce').fillna(0).values
            val_new = pd.to_numeric(df_new[col], errors='coerce').fillna(0).values
            
            # Check length alignment before compare
            min_len = min(len(val_old), len(val_new))
            val_old = val_old[:min_len]
            val_new = val_new[:min_len]

            if np.allclose(val_old, val_new, rtol=1e-05, equal_nan=True):
                 print(f"  [MATCH] Data in '{col}' is identical.")
            else:
                 diff = np.abs(val_old - val_new)
                 max_diff = np.max(diff)
                 print(f"  [MISMATCH] Data in '{col}' differs! Max diff: {max_diff}")
        else:
             print(f"  [SKIP] Cannot compare '{col}' (missing in one file).")

    print("="*60 + "\n")

# --- AUTO RUN ---
if __name__ == "__main__":
    # Check if files exist before running
    if os.path.exists(OLD_FILE_PATH) and os.path.exists(NEW_FILE_PATH):
        compare_logs(OLD_FILE_PATH, NEW_FILE_PATH)
    else:
        print("PLEASE EDIT THE FILE PATHS AT THE TOP OF THE SCRIPT FIRST.")