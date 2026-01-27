# Simple python file to test code for the Ontogeny dataset

# Library:
import os
import numpy as np
import pandas as pd
import pickle
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
print("-" * 30)

# Check if all fish have a .pickle cam_log file
# Prep long-form DataFrame
all_data = []
for root, dirs, files in os.walk(os.path.join(data_folder,protocol_name)):
    # Look for camlog.pkl and camlog.txt files in the same directory
    File_pkl = [f for f in files if f.startswith('OMR_Ontogeny_VOL_') and f.endswith(('.pickle', '.pkl'))]
    File_txt = [f for f in files if f.startswith('OMR_Ontogeny_VOL_') and f.endswith('.txt')]

    if File_txt:
        Raw_filename = File_txt[0]
        Fish_name = Raw_filename.replace('000.txt', '')
        # Parse file name parts ex.: OMR_Ontogeny_VOL_18_03_19_Tu_Tank3627_C20_12dpf_P4_77_66_Atlas000

        # Initialize variables to 'Unknown' so code doesn't crash if Parse fails
        Strain = "Unknown"
        Setup = "Unknown"

        try:
            Filename_parts = Fish_name.split('_')
            # Check length to avoid Index Error
            if len(Filename_parts) == 14:
                Strain = Filename_parts[6]
                Setup = Filename_parts[13]
            else:
                print(f"Warning: Filename format unexpected in {Raw_filename}")

        except Exception as e:
            print(f"Error: no raw .txt file in {root}: {e}")

        pickle = 1 if File_pkl else 0
        all_data.append({
            "Strain": Strain,
            "Fish_ID": Fish_name,
            "Setup": Setup,
            "Pickle_file": pickle
            })

# Convert all_data to DataFrame
All_data_df = pd.DataFrame(all_data)

# Print how many strains there are
unique_strains = All_data_df['Strain'].unique()
print(f"Total number of strains found: {len(unique_strains)}")
print(f"Strains list: {unique_strains}")
print("-" * 30)

# Count total number of Giants and Tu
total_giant = len(All_data_df[(All_data_df['Strain'] == 'Giant')])
total_tu = len(All_data_df[(All_data_df['Strain'] == 'Tu')])
print(f"Total number of fish in the dataset: {total_giant + total_tu}")
print(f"Total N of Giants: {total_giant}; total N of Tu: {total_tu}")

# Count Giant with no pickle files
giant_p0 = len(All_data_df[(All_data_df['Strain'] == 'Giant') & (All_data_df['Pickle_file'] == 0)])
print(f"Giant with no pickle files: {giant_p0}")

# Count Giant with pickle files (pre-processed)
giant_p1 = len(All_data_df[(All_data_df['Strain'] == 'Giant') & (All_data_df['Pickle_file'] == 1)])
print(f"Giant with pickle files (pre-processed): {giant_p1}")

# Count Tu with no pickle files
tu_p0 = len(All_data_df[(All_data_df['Strain'] == 'Tu') & (All_data_df['Pickle_file'] == 0)])
print(f"Tu with no pickle files:    {tu_p0}")

# Count Tu with pickle files (pre-processed)
tu_p1 = len(All_data_df[(All_data_df['Strain'] == 'Tu') & (All_data_df['Pickle_file'] == 1)])
print(f"Tu with pickle files (pre-processed)1:    {tu_p1}")

# Load Fish (using loaders) iteratively
# Select first fish from Tu strain
tu_fish = expe_table[expe_table.strain == 'Tu']
fish_id = tu_fish.index[0]  # Get first valid fish ID
fish = expe_registry.get_recording(fish_id)

print(f"Selected fish ID: {fish_id}")
print(f"Fish experiment name: {fish.expe_name}")

