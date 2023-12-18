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
            flow_sum for flow_sum in statistics[scene]["flow_sums"] if flow_sum > 50000
        ]

    # Define the elements to analyze and their corresponding colors
    elements = [
        "g_means",
        "diff_g_sums",
        "cloud_percents",
        "light_percents",
        "flow_sums",
    ]
    labels = [
        "Mean G Value",
        "G Difference",
        "Cloud Coverage",
        "Light Percentage",
        "Optical Flow Magnitude",
    ]
    colors = ["green", "green", "red", "red", "blue"]

    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(len(elements), 1)

    # Plot each element in a separate subplot
    for i, (element, label, color) in enumerate(zip(elements, labels, colors)):
        ax = plt.subplot(gs[i, 0])

        # Create the boxplot for the current element
        ax.boxplot(
            [statistics[scene][element] for scene in statistics.keys()],
            showfliers=False,
            medianprops={"color": color},
        )
        ax.set_title(f"{label} per Scene")
        ax.set_xticks(np.arange(1, len(statistics) + 1))
        ax.set_xticklabels([f"Scene {i}" for i in range(1, len(statistics) + 1)])

    plt.suptitle(f"Video Statistics Analysis with Frame Step {FRAME_STEP}")
    plt.tight_layout()

    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_part1.png"
    plt.savefig(plot_path)
    print(f"💾 Plot saved at {plot_path}")

    plt.show()


if __name__ == "__main__":
    statistics = np.load(STATISTICS_PATH, allow_pickle=True).item()

    show_plots(statistics)
