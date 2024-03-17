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
    """
    Get the scene number from the filename.

    Args:
        filename (str): The filename

    Returns:
        scene_number (int): The scene number
    """
    return int(filename.split("e")[-1].split(".")[0])


def show_plots(statistics: dict):
    """
    Show plots of the statistics.

    Args:
        statistics (dict): The statistics
    """
    # Sort final statistics by scene number
    filtered_keys = [key for key in statistics.keys() if key.startswith("P1Scene")]
    statistics = {
        k: statistics[k]
        for k in sorted(filtered_keys, key=lambda item: get_scene_number(item))
    }

    # Remove too low values for diff_g_sums and flow_sums
    for scene in statistics.keys():
        statistics[scene]["diff_g_sums"] = [
            diff_g_sum
            for diff_g_sum in statistics[scene]["diff_g_sums"]
            if diff_g_sum > 100000
        ]
        statistics[scene]["flow_sums"] = [
            flow_sum for flow_sum in statistics[scene]["flow_sums"] if flow_sum > 0.03
        ]

    print(statistics['P1Scene01.mp4'].keys())

    # Define the elements to analyze and their corresponding colors
    elements = [
        "diff_g_sums",
        "cloud_percents",
        "light_percents",
        "flow_sums",
    ]
    labels = [
        "Sum of G-Values Differences",
        "Cloud Coverage",
        "Changes in Percentage of Light Pixels",
        "Mean Optical Flow Magnitude",
    ]
    colors = ["green", "green", "red", "red", "blue"]

    # Create a figure with multiple subplots
    plt.figure(figsize=(12, 30))
    gs = gridspec.GridSpec(2 * len(elements) + 1, 1)

    # Set the first subplot about relative deviations in G values
    ax = plt.subplot(gs[0, 0])
    mean_g_values = [statistics[scene]["g_means"] for scene in statistics.keys()]
    relative_deviation_g_values = [
        np.abs((mean_g_value - np.mean(mean_g_value)) / np.mean(mean_g_value)) * 100
        for mean_g_value in mean_g_values 
    ]
    ax.boxplot(
            relative_deviation_g_values,
            showfliers=False,
            medianprops={"color": "green"},
            widths=0.8
        )
    ax.set_title(f"Relative deviations in G-Values (LDR)")
    ax.set_xticks(np.arange(1, len(statistics) + 1))
    ax.set_xticklabels([f"{i}" for i in range(1, len(statistics) + 1)], fontsize=15)
    ax.set_ylim(bottom=0)

    for i, (element, label, color) in enumerate(zip(elements, labels, colors)):
        ax = plt.subplot(gs[i+1, 0])

        # Create the boxplot for the current element
        ax.boxplot(
            [statistics[scene][element] for scene in statistics.keys()],
            showfliers=False,
            medianprops={"color": color},
            widths=0.8
        )
        ax.set_title(f"{label}")
        ax.set_xticks(np.arange(1, len(statistics) + 1))
        ax.set_xticklabels([f"{i}" for i in range(1, len(statistics) + 1)], fontsize=15)
        ax.set_ylim(bottom=0)

    for i, (element, label, color) in enumerate(zip(elements, labels, colors)):
        ax = plt.subplot(gs[i+len(elements)+1, 0])

        # Create the boxplot for the current element
        scene_statistics = [statistics[scene][element] for scene in statistics.keys()]
        relative_deviations = [
            np.abs((scene_statistic - np.mean(scene_statistic)) / np.mean(scene_statistic)) * 100
            for scene_statistic in scene_statistics 
        ]
        ax.boxplot(
            relative_deviations,
            showfliers=False,
            medianprops={"color": color},
            widths=0.8
        )
        ax.set_title(f"{label} (Deviation from mean)")
        ax.set_xticks(np.arange(1, len(statistics) + 1))
        ax.set_xticklabels([f"{i}" for i in range(1, len(statistics) + 1)], fontsize=15)
        ax.set_ylim(bottom=0)

    plt.tight_layout(pad=10.0)

    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_part1.png"
    plt.savefig(plot_path, dpi=500)
    print(f"💾 Plot saved at {plot_path}")

    plt.show()


if __name__ == "__main__":
    statistics = np.load(STATISTICS_PATH, allow_pickle=True).item()

    show_plots(statistics)
