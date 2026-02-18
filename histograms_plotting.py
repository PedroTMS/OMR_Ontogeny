# Code file to make plots from histogram data
"""
This code replicates the original MATLAB plotting logic to generate Mean +/- SEM
histograms for behavioral metrics (Bout Duration, Interbout Interval). It loads
the 'Analysis_All_Histograms.pkl' and 'Analysis_BySpeed_Histograms.pkl' files and
generates comparative figures between Species (Giant vs Tu) across developmental
stages and stimulus speeds. The plotting functions directly mirror the functionality
of 'plot_mean_sem_hist.m' and 'plothistograms_alldata.m'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
DATASET_PATH = Path("dataset")  # Path where .pkl files are saved
RESULTS_PATH = Path("results")  # Path where .png figures will be saved
DATASET_ALL_NAME = 'Analysis_All_Histograms_v2.pkl'
DATASET_BYSPEED_NAME = 'Analysis_BySpeed_Histograms_v2.pkl'
BIN_SIZE = 0.05 # Seconds (Time bin width)
MAX_DURATION = 2.0 # Seconds (Maximum duration analyzed)
BIN_CENTERS = np.arange(0, MAX_DURATION + BIN_SIZE, BIN_SIZE)[:-1] + (BIN_SIZE / 2) # Center of bins (Seconds)

# Plotting Control
# 0 -> plots everything manual
# 1 -> plots everything megabouts
# 2 -> plots for manual and then for megabouts
PLOT_FLAG = 2

# Saving Control
# 0 -> doesn't save anything
# 1 -> save all generated figures as pngs in the Path('results') folder
SAVE_FLAG = 0

# Metric Type Control
# 0 -> 'bout' (Bout Duration)
# 1 -> 'ibi'  (Interbout Interval)
# 2 -> Both
METRIC_TYPE_FLAG = 2

# Comparison Groups (Matches plothistograms_alldata.m)
SPECIES_GROUPS = {
    'Giant': [4, 5, 6, 7, 8, 10], # Ages to plot for Giant (dpf)
    'Tu': [4, 5, 6, 8, 10, 12, 14] # Ages to plot for Tu (dpf)
}

def load_data():
    """Loads the analysis dataframes if they exist."""
    f_all = DATASET_PATH / DATASET_ALL_NAME
    f_speed = DATASET_PATH / DATASET_BYSPEED_NAME
    
    df_all = None
    df_speed = None
    
    if f_all.exists():
        print(f"Loading {f_all}...")
        df_all = pd.read_pickle(f_all)
    else:
        print(f"Warning: {f_all} not found.")

    if f_speed.exists():
        print(f"Loading {f_speed}...")
        df_speed = pd.read_pickle(f_speed)
    else:
        print(f"Warning: {f_speed} not found.")
        
    return df_all, df_speed

def save_figure(fig, filename):
    """Saves the figure if SAVE_FLAG is 1."""
    if SAVE_FLAG == 1:
        if not RESULTS_PATH.exists():
            RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        
        save_path = RESULTS_PATH / f"{filename}.png"
        print(f"Saving figure to {save_path}...")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_mean_sem_hist(ax, df, species, age, metric_type='bout', source='Manual', color=None):
    """
    Replicates plot_mean_sem_hist.m: Plots Mean +/- SEM for a specific group.
    
    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame (Analysis_All_Histograms).
        species: 'Giant' or 'Tu'.
        age: Age in dpf (int).
        metric_type: 'bout' or 'ibi'.
        source: 'Manual' or 'Megabouts'.
        color: Color for the plot line/fill.
    """
    # Filter Data
    subset = df[(df['Species'] == species) & (df['Age'] == age)]
    
    if subset.empty:
        # print(f"No data for {species} {age}dpf")
        return

    # Select Column based on metric and source (e.g., 'Manual_Bout_Prob')
    # Using Probability Density as default (useNormalized=true in MATLAB)
    # [FIX] Handle 'IBI' capitalization explicitly
    metric_label = "IBI" if metric_type.lower() == "ibi" else metric_type.capitalize()
    col_name = f"{source}_{metric_label}_Prob"
    
    if col_name not in subset.columns:
        return

    # Stack arrays (N fish x M bins)
    # The data in the dataframe are lists/arrays, so we stack them into a 2D matrix
    # Unit: Probability Density
    data_matrix = np.stack(subset[col_name].values) # stack arrays to make a matrix
    
    # Calculate Mean and SEM
    # Unit: Probability Density
    mean = np.mean(data_matrix, axis=0)
    sem = np.std(data_matrix, axis=0) / np.sqrt(data_matrix.shape[0])
    n = data_matrix.shape[0] # get number of lines (N fish)
    
    # X-Axis (Bin Centers)
    # [MODIFIED] Convert Seconds to Milliseconds
    x = BIN_CENTERS[:len(mean)] * 1000 
    
    # Plot Shaded SEM
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25, linewidth=0)
    
    # Plot Mean Line
    label = f"{species} {age}dpf (N={n})"
    ax.plot(x, mean, color=color, linewidth=2, label=label)
    
    # Formatting
    ax.set_title(f"{source} {metric_label} Histograms")
    ax.set_ylabel("Probability Density") # Unit: PDF
    ax.set_xlabel("Time (ms)") # [MODIFIED] Unit: Milliseconds
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')

def plot_mean_sem_hist_speed(ax, df, species, age, speed, metric_type='bout', source='Manual', color=None):
    """
    Replicates plot_mean_sem_hist_speed.m: Plots Mean +/- SEM for a specific speed group.

    Args:
        ax: Matplotlib axes to plot on.
        df: DataFrame (Analysis_BySpeed_Histograms).
        species: 'Giant' or 'Tu'.
        age: Age in dpf (int).
        speed: Stimulus speed in mm/s (0, 3, 5, 10, 15, or 30).
        metric_type: 'bout' or 'ibi'.
        source: 'Manual' or 'Megabouts'.
        color: Color for the plot line/fill.
    """
    # Filter Data (Includes Speed)
    # Speed Unit: mm/s
    subset = df[(df['Species'] == species) & (df['Age'] == age) & (np.isclose(df['Speed'], speed))]
    
    if subset.empty:
        return

    # Unit: Normalized Probability Density
    # [FIX] Handle 'IBI' capitalization explicitly
    metric_label = "IBI" if metric_type.lower() == "ibi" else metric_type.capitalize()
    col_name = f"{source}_{metric_label}_Prob"
    
    if col_name not in subset.columns:
        print(f"Warning: Column {col_name} not found.")
        return
    
    data_matrix = np.stack(subset[col_name].values)
    mean = np.mean(data_matrix, axis=0)
    sem = np.std(data_matrix, axis=0) / np.sqrt(data_matrix.shape[0])
    n = data_matrix.shape[0] # number of rows (N fish)
    
    # X-Axis (Bin Centers)
    # [MODIFIED] Convert Seconds to Milliseconds
    x = BIN_CENTERS[:len(mean)] * 1000
    
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.25, linewidth=0)
    label = f"{species} {age}dpf @ {speed} mm/s (N={n})"
    ax.plot(x, mean, color=color, linewidth=2, label=label)
    
    ax.set_title(f"{source} {metric_label} @ {speed} mm/s")
    ax.set_xlabel("Time (ms)") # [MODIFIED] Unit: Milliseconds
    ax.grid(True, alpha=0.3)

def generate_color_palette(n_colors):
    """Generates a list of distinct colors."""
    return sns.color_palette("husl", n_colors)

def replicate_plothistograms_alldata(df_all, df_speed, source='Manual'):
    """
    Replicates the figures generated in plothistograms_alldata.m
    Args:
        source: 'Manual' or 'Megabouts'
    """
    if df_all is None:
        return

    print(f"--- Generating Plots for Source: {source} ---")

    # Determine Metrics to Plot based on Flag
    if METRIC_TYPE_FLAG == 0:
        metrics_to_plot = ['bout']
    elif METRIC_TYPE_FLAG == 1:
        metrics_to_plot = ['ibi']
    else: # Default/Both
        metrics_to_plot = ['bout', 'ibi']

    # --- FIGURE 1 & 2: POOLED BOUTS & IBI (By Age) ---
    for metric in metrics_to_plot:
        metric_label = "IBI" if metric.lower() == "ibi" else metric.capitalize()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        fig.suptitle(f"Pooled {metric_label} Distributions ({source})", fontsize=16)
        
        # [MODIFIED] Set X-Limit: 0.5s -> 500ms (Bouts), 1.0s -> 1000ms (IBI)
        x_limit = 500 if metric == 'bout' else 1000

        # Giant Subplot
        colors_g = generate_color_palette(len(SPECIES_GROUPS['Giant']))
        for i, age in enumerate(SPECIES_GROUPS['Giant']):
            plot_mean_sem_hist(axes[0], df_all, 'Giant', age, metric, source, colors_g[i])
        axes[0].set_title("Giant")
        axes[0].set_xlim(0, x_limit) 

        # Tu Subplot
        colors_t = generate_color_palette(len(SPECIES_GROUPS['Tu']))
        for i, age in enumerate(SPECIES_GROUPS['Tu']):
            plot_mean_sem_hist(axes[1], df_all, 'Tu', age, metric, source, colors_t[i])
        axes[1].set_title("Tu")
        axes[1].set_xlim(0, x_limit) 
        
        # Save Figure
        save_figure(fig, f"Pooled_{metric_label}_{source}")

    # --- FIGURE 3 & 4: BY SPEED (Speed 0 Comparison) ---
    if df_speed is not None:
        for metric in metrics_to_plot:
            metric_label = "IBI" if metric.lower() == "ibi" else metric.capitalize()
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
            fig.suptitle(f"{metric_label} Distributions at 0 mm/s ({source})", fontsize=16)
            
            # [MODIFIED] Set X-Limit: 0.5s -> 500ms (Bouts), 1.0s -> 1000ms (IBI)
            x_limit = 500 if metric == 'bout' else 1000
            
            # Giant Speed 0
            colors_g = generate_color_palette(len(SPECIES_GROUPS['Giant']))
            for i, age in enumerate(SPECIES_GROUPS['Giant']):
                plot_mean_sem_hist_speed(axes[0], df_speed, 'Giant', age, 0, metric, source, colors_g[i])
            axes[0].set_title("Giant (0 mm/s)")
            axes[0].set_xlim(0, x_limit)
            
            # Tu Speed 0
            colors_t = generate_color_palette(len(SPECIES_GROUPS['Tu']))
            for i, age in enumerate(SPECIES_GROUPS['Tu']):
                plot_mean_sem_hist_speed(axes[1], df_speed, 'Tu', age, 0, metric, source, colors_t[i])
            axes[1].set_title("Tu (0 mm/s)")
            axes[1].set_xlim(0, x_limit)
            
            # Save Figure
            save_figure(fig, f"Speed0_{metric_label}_{source}")

    # --- FIGURE 5: GRID PLOT (All Speeds) ---
    if df_speed is not None:
        speeds = [0, 3, 5, 10, 15, 30] # Units: mm/s
        
        # Now loops through metrics (Dynamic Grid)
        for metric in metrics_to_plot:
            metric_label = "IBI" if metric.lower() == "ibi" else metric.capitalize()
            
            fig, axes = plt.subplots(2, 6, figsize=(20, 8), constrained_layout=True, sharex=True, sharey='row')
            fig.suptitle(f"{metric_label} Duration Across All Speeds ({source})", fontsize=16)
            
            # [MODIFIED] Set X-Limit: 0.5s -> 500ms (Bouts), 1.0s -> 1000ms (IBI)
            x_limit = 500 if metric == 'bout' else 1000
            
            # Row 0: Giant
            colors_g = generate_color_palette(len(SPECIES_GROUPS['Giant']))
            for col, speed in enumerate(speeds):
                ax = axes[0, col]
                for i, age in enumerate(SPECIES_GROUPS['Giant']):
                    plot_mean_sem_hist_speed(ax, df_speed, 'Giant', age, speed, metric, source, colors_g[i])
                ax.set_title(f"Giant @ {speed} mm/s")
                ax.set_xlim(0, x_limit) 
                if col == 0:
                    ax.legend(fontsize='xx-small')

            # Row 1: Tu
            colors_t = generate_color_palette(len(SPECIES_GROUPS['Tu']))
            for col, speed in enumerate(speeds):
                ax = axes[1, col]
                for i, age in enumerate(SPECIES_GROUPS['Tu']):
                    plot_mean_sem_hist_speed(ax, df_speed, 'Tu', age, speed, metric, source, colors_t[i])
                ax.set_title(f"Tu @ {speed} mm/s")
                ax.set_xlim(0, x_limit) 
                
            # Save Figure
            save_figure(fig, f"Grid_AllSpeeds_{metric_label}_{source}")
            
    plt.show()

def main():
    # 1. Load Data
    df_all, df_speed = load_data()
    
    if df_all is None:
        print("Cannot proceed without main histogram data.")
        return

    # 2. Generate Plots based on Flag
    # PLOT_FLAG defined in CONFIGURATION: 0=Manual, 1=Megabouts, 2=Both
    
    if PLOT_FLAG == 0:
        # Plot Manual Only
        replicate_plothistograms_alldata(df_all, df_speed, source='Manual')
        
    elif PLOT_FLAG == 1:
        # Plot Megabouts Only
        replicate_plothistograms_alldata(df_all, df_speed, source='Megabouts')
        
    elif PLOT_FLAG == 2:
        # Plot Manual then Megabouts
        replicate_plothistograms_alldata(df_all, df_speed, source='Manual')
        replicate_plothistograms_alldata(df_all, df_speed, source='Megabouts')
    
    else:
        print(f"Invalid PLOT_FLAG: {PLOT_FLAG}. Use 0 (Manual), 1 (Megabouts), or 2 (Both).")
    
    print("Done.")

if __name__ == "__main__":
    main()