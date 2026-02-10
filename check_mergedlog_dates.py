import os
import datetime
import pandas as pd
import config  # Imports your existing config file for ROOT_PATH

def get_file_info(file_path):
    """
    Helper to get modification year and full date string.
    Returns (None, None) if file doesn't exist.
    """
    if not os.path.exists(file_path):
        return None, None
        
    timestamp = os.path.getmtime(file_path)
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    return dt_object.year, dt_object.strftime("%Y-%m-%d %H:%M:%S")

def audit_file_dates(root_folder):
    """
    Scans for _MergedLog.pickle files and their corresponding .mat sources.
    Returns a DataFrame comparing their modification dates.
    """
    print(f"Scanning {root_folder} for file audit...")
    print("-" * 60)
    
    records = []
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(root_folder):
        
        # 1. Identify Merged Logs in this folder
        merged_files = [f for f in files if f.endswith('_MergedLog.pickle')]
        
        for merged_f in merged_files:
            base_name = merged_f.replace('_MergedLog.pickle', '')
            
            # Paths
            merged_path = os.path.join(root, merged_f)
            mat_name = base_name + '.mat' # Assumes standard naming convention
            mat_path = os.path.join(root, mat_name)
            
            # Get Dates
            merged_year, merged_date = get_file_info(merged_path)
            mat_year, mat_date = get_file_info(mat_path)
            
            # Logic: Flag suspicious cases
            # Case A: Merged exists but Raw is missing (Very bad)
            if mat_year is None:
                status = "MISSING RAW"
            # Case B: Merged is old (2019) but Raw is new (2026) -> Risk of mismatch
            elif merged_year < 2021 and mat_year > 2025:
                status = "MISMATCH (Old Log/New Raw)"
            # Case C: Both old
            elif merged_year < 2021 and mat_year < 2021:
                status = "CONSISTENT (Old)"
            # Case D: Both new
            elif merged_year > 2025 and mat_year > 2025:
                status = "CONSISTENT (New)"
            else:
                status = "MIXED"

            records.append({
                'filename': base_name,
                'status': status,
                'merged_year': merged_year,
                'raw_year': mat_year,
                'merged_date': merged_date,
                'raw_date': mat_date,
                'folder': root
            })

    if not records:
        print("No files found to audit.")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(records)
    
    # --- REPORT ---
    print(f"\nTotal Files Audited: {len(df)}")
    
    # Count by Status
    print("\nStatus Breakdown:")
    print(df['status'].value_counts())
    
    print("-" * 60)
    
    # Highlight specific risks
    mismatches = df[df['status'] == "MISMATCH (Old Log/New Raw)"]
    if not mismatches.empty:
        print(f"\n[WARNING] Found {len(mismatches)} files where MergedLog is OLD but .mat is NEW.")
        print("These should likely be re-processed to ensure they match the current raw data.")
    
    return df

# --- EXECUTION ---
if __name__ == "__main__":
    df_audit = audit_file_dates(config.ROOT_PATH)
    
    # Display a sample of the table
    if not df_audit.empty:
        print("\nSample Data (Head):")
        cols = ['filename', 'status', 'merged_year', 'raw_year']
        print(df_audit[cols].head(10).to_string())
        
        # Save for Excel review
        # df_audit.to_csv("Audit_Report.csv", index=False)