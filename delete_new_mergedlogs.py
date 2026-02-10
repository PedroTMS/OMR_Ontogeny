# Simple code to find recently generated _MergedLog.pkl files (from 2026),
# and safely get rid of them.
"""
This code lists all _MergedLog.pkl files, audits their date,
selects the recently generated ones (from 2026), ask the user
for confirmation and, if yes, safely eliminates them.

Type DELETE when prompted, if you agree.
"""

import os
import datetime
import config  # Imports your ROOT_PATH

def delete_new_merged_logs(root_folder):
    print(f"Scanning {root_folder} for 2026 Merged Logs...")
    print("-" * 60)
    
    files_to_delete = []
    
    # 1. Scan and Collect
    for root, dirs, files in os.walk(root_folder):
        for f in files:
            if f.endswith('_MergedLog.pickle'):
                full_path = os.path.join(root, f)
                
                # Check modification time
                timestamp = os.path.getmtime(full_path)
                dt_object = datetime.datetime.fromtimestamp(timestamp)
                
                # Condition: strictly from year 2026
                if dt_object.year == 2026:
                    files_to_delete.append(full_path)

    if not files_to_delete:
        print("No 2026 merged logs found! (You are safe)")
        return

    # 2. Report
    print(f"Found {len(files_to_delete)} files from 2026:")
    for path in files_to_delete[:10]: # Print first 10 as sample
        print(f"  - {os.path.basename(path)}")
    if len(files_to_delete) > 10:
        print(f"  ... and {len(files_to_delete) - 10} more.")
        
    print("-" * 60)
    print("These files contain the OLD schema (missing stim_iter) and must be deleted.")
    
    # 3. Confirm Delete
    confirm = input("Type 'DELETE' to permanently remove these files: ")
    
    if confirm == 'DELETE':
        print("\nDeleting...")
        count = 0
        for path in files_to_delete:
            try:
                os.remove(path)
                count += 1
            except Exception as e:
                print(f"Error deleting {path}: {e}")
        print(f"\n[SUCCESS] Deleted {count} files.")
        print("You can now run main.py to regenerate them correctly.")
    else:
        print("\n[CANCELLED] No files were touched.")

if __name__ == "__main__":
    # Ensure config path is valid
    if os.path.exists(config.ROOT_PATH):
        delete_new_merged_logs(config.ROOT_PATH)
    else:
        print("Error: ROOT_PATH in config.py is invalid.")