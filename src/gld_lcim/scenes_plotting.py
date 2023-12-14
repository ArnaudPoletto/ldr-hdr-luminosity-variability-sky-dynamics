import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
GENERATED_PATH = str(GLOBAL_DIR / "generated") + "/"
GLD_LCIM_PATH = DATA_PATH + f"gld_lcim/"
STATISTICS_PATH = GLD_LCIM_PATH + "statistics.npy"

FRAME_STEP = 75

def get_scene_number(filename):
    if 'Clear' in filename:
        return int(filename.split('Clear')[-1].split('.')[0])
    elif 'Overcast' in filename:
        return int(filename.split('Overcast')[-1].split('.')[0])
    else:
        return int(filename.split('e')[-1].split('.')[0])
    

def show_plots_part2(statistics):
    clear_keys = sorted([key for key in statistics.keys() if key.startswith('P2Clear')], key=lambda item: get_scene_number(item))
    overcast_keys = sorted([key for key in statistics.keys() if key.startswith('P2Overcast')], key=lambda item: get_scene_number(item))

    # Ensure that we have matching 'Clear' and 'Overcast' scenes
    paired_keys = list(zip(clear_keys, overcast_keys))

    elements = ['g_means', 'diff_g_sums', 'cloud_percents', 'light_percents', 'flow_sums']
    labels = ['Mean G Value', 'G Difference', 'Cloud Coverage', 'Light Percentage', 'Optical Flow Magnitude']
    colors = ['green', 'green', 'red', 'red', 'blue']

    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(len(elements), 1)

    for i, (element, label, color) in enumerate(zip(elements, labels, colors)):
        ax = plt.subplot(gs[i, 0])

        # We'll create pairs of data for 'Clear' and 'Overcast' to plot side by side
        data_pairs = []
        positions = []
        for j, (clear_key, overcast_key) in enumerate(paired_keys):
            data_pairs.append(statistics[clear_key][element])
            data_pairs.append(statistics[overcast_key][element])
            positions.extend([j * 4 + 1.25, j * 4 + 2.75])  # Position the pairs with a space for the scene label

        # Plot 'Clear' and 'Overcast' side by side
        ax.boxplot(data_pairs, positions=positions, widths=0.6, showfliers=False, medianprops={'color': color})

        # Set the primary tick labels for 'Clear' and 'Overcast'
        ax.set_xticks(positions)
        ax.set_xticklabels(['Clear', 'Overcast'] * len(paired_keys))

        # Add a secondary axis for scene numbering
        secax = ax.secondary_xaxis('top')
        ax.set_title(f"{label} per Scene")
        secax.set_xticks([1.75 + j * 4 for j in range(len(paired_keys))])
        secax.set_xticklabels([f"Scene {j+1}" for j in range(len(paired_keys))])

        # Minor grid for visual separation
        ax.xaxis.grid(True, which='minor', linestyle='', linewidth='0.5', color='gray')
        # Major grid for scene separation
        ax.xaxis.grid(True, which='major', linestyle='', linewidth='0.5', color='black')

    plt.suptitle(f"Video Statistics Analysis (Part 2) with Frame Step {FRAME_STEP}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to prevent overlap

    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_part2.png"
    plt.savefig(plot_path)
    print(f'💾 Plot saved at {plot_path}')

    plt.show()


def show_plots(statistics: dict, part: int = 1):
    if part not in [1, 2, 3]:
        raise Exception(f"❌ Invalid part {part}: must be 1, 2 or 3.")
    
    if part == 2:
        show_plots_part2(statistics)
        return
    
    # Sort statistics by taking the ones of the form PX...
    if part == 1:
        filtered_keys = [key for key in statistics.keys() if key.startswith('P1Scene')]
    elif part == 2:
        filtered_keys = [key for key in statistics.keys() if key.startswith('P2Clear') or key.startswith('P2Overcast')]
    else:  # part == 3
        filtered_keys = [key for key in statistics.keys() if key.startswith('P3Scene')]
    
    # Sort final statistics by scene number
    statistics = {k: statistics[k] for k in sorted(filtered_keys, key=lambda item: get_scene_number(item))}

    # Remove too low values for diff_g_sums and flow_sums
    for scene in statistics.keys():
        statistics[scene]['diff_g_sums'] = [diff_g_sum for diff_g_sum in statistics[scene]['diff_g_sums'] if diff_g_sum > 100000]
        statistics[scene]['flow_sums'] = [flow_sum for flow_sum in statistics[scene]['flow_sums'] if flow_sum > 50000]

    # Define the elements to analyze and their corresponding colors
    elements = ['g_means', 'diff_g_sums', 'cloud_percents', 'light_percents', 'flow_sums']
    labels = ['Mean G Value', 'G Difference', 'Cloud Coverage', 'Light Percentage', 'Optical Flow Magnitude']
    colors = ['green', 'green', 'red', 'red', 'blue']

    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(len(elements), 1)

    # Plot each element in a separate subplot
    for i, (element, label, color) in enumerate(zip(elements, labels, colors)):
        ax = plt.subplot(gs[i, 0])
        
        # Create the boxplot for the current element
        ax.boxplot([statistics[scene][element] for scene in statistics.keys()], showfliers=False, medianprops={'color': color})
        ax.set_title(f"{label} per Scene")
        ax.set_xticks(np.arange(1, len(statistics) + 1))
        ax.set_xticklabels([f"Scene {i}" for i in range(1, len(statistics) + 1)])

    plt.suptitle(f"Video Statistics Analysis with Frame Step {FRAME_STEP}")
    plt.tight_layout()

    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_part{part}.png"
    plt.savefig(plot_path)
    print(f'💾 Plot saved at {plot_path}')

    plt.show()


if __name__ == '__main__':
    statistics = np.load(STATISTICS_PATH, allow_pickle=True).item()

    for i in range(1, 4):
        show_plots(statistics, part=i)