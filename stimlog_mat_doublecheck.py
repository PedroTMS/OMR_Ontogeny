# Simple code to walk the root, find all stimlog.mat files and check when they were last modified
"""
Importat safety check on stimlog mat files. Checkes
whether the files were made and last modified in 2019
or if they were modified recently in 2016.
"""

import os
import datetime
import pandas as pd

def check_file_dates(root_folder):
    print(f"Scanning {root_folder} for Stimulus Logs...")
    print("-" * 60)
    
    records = []
    
    for root, dirs, files in os.walk(root_folder):
        # Find files that look like stimlogs (case insensitive)
        stim_files = [f for f in files if 'stimlog' in f.lower() and f.endswith('.mat')]
        
        for f in stim_files:
            full_path = os.path.join(root, f)
            
            # Get last modified timestamp
            timestamp = os.path.getmtime(full_path)
            dt_object = datetime.datetime.fromtimestamp(timestamp)
            
            records.append({
                'filename': f,
                'folder': root,
                'year': dt_object.year,
                'full_date': dt_object.strftime("%Y-%m-%d %H:%M:%S")
            })

    if not records:
        print("No stimlog.mat files found!")
        return

    # Create a DataFrame for analysis
    df = pd.DataFrame(records)
    
    # --- REPORT ---
    print(f"\nTotal Stimlogs Found: {len(df)}")
    
    # Group by Year
    year_counts = df['year'].value_counts().sort_index()
    print("\nFile Vintage Breakdown:")
    for year, count in year_counts.items():
        print(f"  Year {year}: {count} files")
        
    # Check for mixture
    years_found = df['year'].unique()
    
    print("-" * 60)
    if len(years_found) > 1:
        print("⚠️  WARNING: MIXED DATA DETECTED!")
        print("You have both old and new files. This suggests some might have been re-saved.")
        
        # Optional: Show a few examples of the 'New' files
        newest_year = max(years_found)
        print(f"\nExamples of recent files ({newest_year}):")
        recent_files = df[df['year'] == newest_year].head(5)
        for _, row in recent_files.iterrows():
            print(f"  - {row['filename']} (Modified: {row['full_date']})")
    else:
        print(f"✅  CONSISTENT: All files are from {years_found[0]}.")
        
    print("-" * 60)

# --- EXECUTION ---
if __name__ == "__main__":
    # Update this path to your actual data drive
    ROOT_PATH = 'F:/OMR_Ontogeny_VOL' 
    
    check_file_dates(ROOT_PATH)