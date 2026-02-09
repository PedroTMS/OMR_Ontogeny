# Main code file
"""
This script is the entry point for the merging of critical data from
camlog and stimlog mat files into single MergedLog.pkl files for
each fish of the Ontogeny OMR experimental data. It initializes
the database, identifies unprocessed recordings, and triggers the
batch processing loop. It handles global error tracking and
provides a final summary report.
"""

import time
import pandas as pd
import config
import io_utils
import processing

def main():
    """
    Main execution function.
    """
    # Start timer for performance tracking
    start_time = time.time()
    
    ### Build the Database ###
    # Scans the root folder defined in config.py for experiments
    print(f"Initializing pipeline...")
    df_experiments = io_utils.generate_database(config.ROOT_PATH)
    
    if df_experiments.empty:
        print(f"[Error] No experiments found in: {config.ROOT_PATH}")
        print("Please check the path in 'config.py'.")
        return

    ### Identify Missing Logs ###
    # Filter for rows where 'saving_flag' is False (files not yet processed)
    to_process = df_experiments[df_experiments['saving_flag'] == False]
    
    total_files = len(df_experiments)
    missing_files = len(to_process)
    processed_files = total_files - missing_files
    
    # Print Database Summary
    print(f"\n{'='*50}")
    print(f"DATABASE SUMMARY")
    print(f"{'='*50}")
    print(f"  Root Path           : {config.ROOT_PATH}")
    print(f"  Total Recordings    : {total_files}")
    print(f"  Already Processed   : {processed_files}")
    print(f"  Pending Processing  : {missing_files}")
    print(f"{'='*50}\n")
    
    if missing_files == 0:
        print("All files are up to date! Exiting.")
        return

    ### Batch Processing Loop ###
    # 
    print(f"Starting batch processing of {missing_files} files...\n")
    
    success_count = 0
    fail_count = 0
    
    # Iterate through the list of pending files
    for idx, row in to_process.iterrows():
        try:
            # Pass the single row (Series) to the processing module
            success = processing.process_recording(row)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                
        except KeyboardInterrupt:
            # Allows user to stop the script safely with Ctrl+C
            print("\n" + "!"*50)
            print("[USER ABORT] Processing stopped by user.")
            print("!"*50 + "\n")
            break
            
        except Exception as e:
            # Catch-all for unexpected crashes to prevent stopping the whole batch
            print(f"  [CRITICAL FAIL] {row['fish_filename']}: {e}")
            fail_count += 1
            
    ### Final Report ###
    # 
    elapsed_seconds = time.time() - start_time
    elapsed_minutes = elapsed_seconds / 60
    
    print(f"\n{'='*50}")
    print(f"BATCH COMPLETE")
    print(f"{'='*50}")
    print(f"  Time Elapsed   : {elapsed_minutes:.2f} minutes")
    print(f"  Successful     : {success_count}")
    print(f"  Failed         : {fail_count}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()