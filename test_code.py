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

# Check if all fish have a .pickle cam_log file
# Prep long-form DataFrame
all_data = []
for root, dirs, files in os.walk(os.path.join(data_folder,protocol_name)):
    # Look for camlog.pkl and camlog.txt files in the same directory
    File_pkl = [f for f in files if f.startswith('OMR_Ontogeny_VOL_') and f.endswith(('.pickle', '.pkl'))]
    File_txt = [f for f in files if f.startswith('OMR_Ontogeny_VOL_') and f.endswith('.txt')]
    if File_txt:
        Filename = File_txt[0]
        # Parse file name parts ex.: OMR_Ontogeny_VOL_18_03_19_Tu_Tank3627_C20_12dpf_P4_77_66_Atlas000
        try:
            File_name_parts = Filename.split('_')
            strain = File_name_parts[7]
        except Exception as e:
            print(f"Error: no raw .txt file in {root}: {e}")

    if File_txt and File_pkl:
        pickle = 1
        all_data.append({
            "Strain": strain,
            "Fish_ID": Filename[:-4],
            "Pickle_file": pickle
            })
    else:
        pickle = 0
        all_data.append({
            "Strain": strain,
            "Fish_ID": Filename[:-4],
            "Pickle_file": pickle
            })

#   print("Directory path: %s"%root)
#   print("Directory Names: %s"%dirs)
#   print("Files Names: %s"%files)

# Load Fish (using loaders) iteratively
# Select first fish from Tu strain
tu_fish = expe_table[expe_table.strain == 'Tu']
fish_id = tu_fish.index[0]  # Get first valid fish ID
fish = expe_registry.get_recording(fish_id)

print(f"Selected fish ID: {fish_id}")
print(f"Fish experiment name: {fish.expe_name}")

# file = "F:\\OMR_Ontogeny_VOL\\atlas\\Giant_Danio\\4dpf\\P1\\OMR_Ontogeny_VOL_27_02_19_Giant_Tank2_C10_04dpf_P1_75_66_Atlas000_MergedLog.pickle"
# with open(file, 'rb') as f:
#     cam_log = pickle.load(f)
# cam_log.columns

# import pandas as pd
# import os

# output_folder = r"F:\\OMR_Ontogeny_VOL\\atlas\\Tu\\10dpf\\P1" 
# file_name = "OMR_Ontogeny_VOL_02_03_19_Tu_Tank3389_C01_10dpf_P1_78_66_Atlas000_MergedLog.pkl"
# # Combine folder and filename
# full_path = os.path.join(output_folder, file_name)
# # Check if the folder exists; if not, create it automatically
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#     print(f"Created new directory: {output_folder}")

# # Save the DataFrame
# try:
#     merged_log.to_pickle(full_path)
#     print(f"Success! Dataframe saved to:\n{full_path}")
# except Exception as e:
#     print(f"Error saving file: {e}")