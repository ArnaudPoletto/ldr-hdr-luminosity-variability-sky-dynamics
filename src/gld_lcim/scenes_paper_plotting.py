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


def get_scene_number(filename, part):
    """
    Get the scene number from the filename.

    Args:
        filename (str): The filename
        part (str): The part (P1, P2, or P3)

    Returns:
        scene_number (int): The scene number
    """
    if part == "P2":
        if "Overcast" in filename:
            return int(filename.split("Overcast")[-1].split(".")[0])
        else:
            return int(filename.split("Clear")[-1].split(".")[0])
    else:
        return int(filename.split("e")[-1].split(".")[0])


def create_violin_plots(statistics: dict):
    """
    Create violin plots for P2 showing distributions of metrics for overcast vs clear conditions.
    """
    filtered_keys = [key for key in statistics.keys() if key.startswith("P2")]

    # Separate overcast and clear scenes
    overcast_keys = [k for k in filtered_keys if "Overcast" in k]
    clear_keys = [k for k in filtered_keys if "Clear" in k]

    # Sort both sets by scene number
    overcast_stats = {
        k: statistics[k]
        for k in sorted(overcast_keys, key=lambda x: get_scene_number(x, "P2"))
    }
    clear_stats = {
        k: statistics[k]
        for k in sorted(clear_keys, key=lambda x: get_scene_number(x, "P2"))
    }
    
    # Get relative deviations in G values for each scene
    for scene in overcast_stats.keys():
        overcast_stats[scene]["rel_dev_g_means"] = np.abs(
            (np.array(overcast_stats[scene]["g_means"] - np.mean(overcast_stats[scene]["g_means"]))) / (np.mean(overcast_stats[scene]["g_means"]) + 1e-6) * 100
        ).tolist()
    for scene in clear_stats.keys():
        clear_stats[scene]["rel_dev_g_means"] = np.abs(
            (np.array(clear_stats[scene]["g_means"]) - np.mean(clear_stats[scene]["g_means"])) / (np.mean(clear_stats[scene]["g_means"]) + 1e-6) * 100
        ).tolist()

    elements = [
        "g_means",
        "rel_dev_g_means",
        "diff_g_sums",
        "cloud_percents",
        "light_percents",
        "flow_sums",
    ]
    labels = [
        "Mean G-Values (LDR)",
        "Relative Deviation in G-Values (LDR)",
        "Sum of G-Values Differences",
        "Cloud Coverage",
        "Changes in Percentage of Light Pixels",
        "Mean Optical Flow Magnitude",
    ]
    colors = ["green", "green", "green", "green", "red", "red"]

    fig, axes = plt.subplots(len(elements), 1, figsize=(15, 4 * len(elements)))
    fig.suptitle(
        "P2 Distribution Comparison: Overcast vs Clear Conditions", fontsize=16, y=0.99
    )

    for idx, (element, label, color) in enumerate(zip(elements, labels, colors)):
        ax = axes[idx]

        # Collect all data for overcast and clear conditions
        overcast_data = []
        clear_data = []

        for scene in overcast_stats.keys():
            data = overcast_stats[scene][element]
            if isinstance(data, list):
                overcast_data.extend(data)
            else:
                overcast_data.append(data)

        for scene in clear_stats.keys():
            data = clear_stats[scene][element]
            if isinstance(data, list):
                clear_data.extend(data)
            else:
                clear_data.append(data)

        # Clean data if needed
        if element in ["diff_g_sums", "flow_sums"]:
            overcast_data = [
                x
                for x in overcast_data
                if x > (0.2 if element == "diff_g_sums" else 0.03)
            ]
            clear_data = [
                x for x in clear_data if x > (0.2 if element == "diff_g_sums" else 0.03)
            ]

        # Create violin plot
        parts = ax.violinplot(
            [overcast_data, clear_data],
            showmeans=False,
            showmedians=True,
            showextrema=False,
            quantiles=None,
        )
        parts["cmedians"].set_color("black")

        # Customize violin plot colors
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)

        ax.set_title(label)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Overcast", "Clear"])

        # Add statistical information
        overcast_mean = np.mean(overcast_data)
        clear_mean = np.mean(clear_data)
        ax.text(
            0.02,
            0.98,
            f"Overcast Mean: {overcast_mean:.3f}\nClear Mean: {clear_mean:.3f}\n"
            f"Difference: {abs(overcast_mean - clear_mean):.3f} ({abs(overcast_mean - clear_mean)/overcast_mean*100:.1f}%)",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    return fig


def create_single_part_plots(statistics: dict, part: str):
    """
    Create plots for P1, P2, or P3.
    """
    filtered_keys = [key for key in statistics.keys() if key.startswith(part)]

    if part == "P2":
        # Separate overcast and clear scenes
        overcast_keys = [k for k in filtered_keys if "Overcast" in k]
        clear_keys = [k for k in filtered_keys if "Clear" in k]

        # Sort both sets by scene number
        overcast_stats = {
            k: statistics[k]
            for k in sorted(overcast_keys, key=lambda x: get_scene_number(x, part))
        }
        clear_stats = {
            k: statistics[k]
            for k in sorted(clear_keys, key=lambda x: get_scene_number(x, part))
        }

        sorted_overcast_keys = {
            k: statistics[k]
            for k in sorted(
                overcast_keys, key=lambda item: get_scene_number(item, part)
            )
        }
        sorted_clear_keys = {
            k: statistics[k]
            for k in sorted(clear_keys, key=lambda item: get_scene_number(item, part))
        }

        # Number of unique scenes
        n_scenes = len(sorted_overcast_keys)
    else:
        sorted_stats = {
            k: statistics[k]
            for k in sorted(
                filtered_keys, key=lambda item: get_scene_number(item, part)
            )
        }

    # Clean data
    if part == "P2":
        for scene in sorted_overcast_keys.keys():
            sorted_overcast_keys[scene]["diff_g_sums"] = [
                x for x in sorted_overcast_keys[scene]["diff_g_sums"] if x > 0.2
            ]
            sorted_overcast_keys[scene]["flow_sums"] = [
                x for x in sorted_overcast_keys[scene]["flow_sums"] if x > 0.03
            ]
        for scene in sorted_clear_keys.keys():
            sorted_clear_keys[scene]["diff_g_sums"] = [
                x for x in sorted_clear_keys[scene]["diff_g_sums"] if x > 0.2
            ]
            sorted_clear_keys[scene]["flow_sums"] = [
                x for x in sorted_clear_keys[scene]["flow_sums"] if x > 0.03
            ]
    else:
        for scene in sorted_stats.keys():
            sorted_stats[scene]["diff_g_sums"] = [
                x for x in sorted_stats[scene]["diff_g_sums"] if x > 0.2
            ]
            sorted_stats[scene]["flow_sums"] = [
                x for x in sorted_stats[scene]["flow_sums"] if x > 0.03
            ]

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
    colors = ["green", "green", "red", "red"]

    fig = plt.figure(figsize=(9, 40) if part == "P2" else (9, 40))
    fig.suptitle(f"Statistics for {part}", fontsize=16)
    gs = gridspec.GridSpec(2 * len(elements) + 2, 1)

    def create_grouped_boxplot(ax, data_overcast, data_clear, color):
        """Helper function to create grouped boxplots for P2"""
        n_scenes = len(data_overcast)
        positions = []
        labels = []
        boxplot_data = []

        for i in range(n_scenes):
            # Calculate positions for this pair
            base_pos = i * 3  # Use 3 units per scene to create spacing
            positions.extend([base_pos + 0.7, base_pos + 1.3])  # Offset within group
            labels.extend(["O", "C"])  # Short labels for Overcast and Clear

            boxplot_data.append(data_overcast[i])
            boxplot_data.append(data_clear[i])

        # Create boxplots
        bp = ax.boxplot(
            boxplot_data,
            positions=positions,
            showfliers=False,
            medianprops={"color": color},
            widths=0.4,
        )

        # Set x-ticks and labels
        scene_positions = [i * 3 + 1 for i in range(n_scenes)]  # Center of each pair
        ax.set_xticks(scene_positions)
        ax.set_xticklabels([f"Scene {i+1}" for i in range(n_scenes)], fontsize=12)

        # Add secondary x-ticks for Overcast/Clear labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, fontsize=8)

        return ax.get_ylim()  # Return ylim for consistency across subplots

    # Plot relative deviations in G values
    if part == "P2":
        ax = fig.add_subplot(gs[0, 0])
        mean_g_values_overcast = [
            overcast_stats[scene]["g_means"] for scene in overcast_stats.keys()
        ]
        mean_g_values_clear = [
            clear_stats[scene]["g_means"] for scene in clear_stats.keys()
        ]
        create_grouped_boxplot(ax, mean_g_values_overcast, mean_g_values_clear, "green")
        ax.set_title(f"Mean G-Values (LDR)")

        ax = fig.add_subplot(gs[1, 0])
        relative_mean_g_values_overcast = [
            np.abs((np.array(scene) - np.mean(scene)) / (np.mean(scene) + 1e-6)) * 100
            for scene in mean_g_values_overcast
        ]
        relative_mean_g_values_clear = [
            np.abs((np.array(scene) - np.mean(scene)) / (np.mean(scene) + 1e-6)) * 100
            for scene in mean_g_values_clear
        ]
        create_grouped_boxplot(
            ax, relative_mean_g_values_overcast, relative_mean_g_values_clear, "green"
        )
        ax.set_title(f"Relative Deviation in G-Values (LDR)")

    else:
        ax = fig.add_subplot(gs[0, 0])
        mean_g_values = [
            sorted_stats[scene]["g_means"] for scene in sorted_stats.keys()
        ]
        ax.boxplot(
            mean_g_values,
            showfliers=False,
            medianprops={"color": "green"},
            widths=0.8,
        )
        ax.set_xticks(np.arange(1, len(sorted_stats) + 1))
        ax.set_xticklabels(
            [f"{i}" for i in range(1, len(sorted_stats) + 1)], fontsize=12
        )
        ax.set_title(f"Mean G-Values (LDR)")

        ax = fig.add_subplot(gs[1, 0])
        relative_mean_g_values = [
            np.abs((np.array(scene) - np.mean(scene)) / (np.mean(scene) + 1e-6)) * 100
            for scene in mean_g_values
        ]
        ax.boxplot(
            relative_mean_g_values,
            showfliers=False,
            medianprops={"color": "green"},
            widths=0.8,
        )
        ax.set_xticks(np.arange(1, len(sorted_stats) + 1))
        ax.set_xticklabels(
            [f"{i}" for i in range(1, len(sorted_stats) + 1)], fontsize=12
        )
        ax.set_title(f"Relative Deviation in G-Values (LDR)")

    for i, (element, label, color) in enumerate(zip(elements, labels, colors)):
        # Raw values
        ax = fig.add_subplot(gs[i + 2, 0])
        if part == "P2":
            data_overcast = [
                overcast_stats[scene][element] for scene in overcast_stats.keys()
            ]
            data_clear = [clear_stats[scene][element] for scene in clear_stats.keys()]
            ylim = create_grouped_boxplot(ax, data_overcast, data_clear, color)
        else:
            ax.boxplot(
                [sorted_stats[scene][element] for scene in sorted_stats.keys()],
                showfliers=False,
                medianprops={"color": color},
                widths=0.8,
            )
            ax.set_xticks(np.arange(1, len(sorted_stats) + 1))
            ax.set_xticklabels(
                [f"{i}" for i in range(1, len(sorted_stats) + 1)], fontsize=12
            )

        ax.set_title(f"{label}")
        # ax.set_ylim(bottom=0)

        # Relative deviations
        ax = fig.add_subplot(gs[i + len(elements) + 2, 0])
        if part == "P2":
            # Calculate relative deviations
            rel_dev_overcast = []
            rel_dev_clear = []

            for scene_idx in range(n_scenes):
                overcast_data = data_overcast[scene_idx]
                clear_data = data_clear[scene_idx]

                rel_dev_overcast.append(
                    np.abs(
                        (np.array(overcast_data) - np.mean(overcast_data))
                        / (np.mean(overcast_data) + 1e-6)
                    )
                    * 100
                )
                rel_dev_clear.append(
                    np.abs(
                        (np.array(clear_data) - np.mean(clear_data))
                        / (np.mean(clear_data) + 1e-6)
                    )
                    * 100
                )

            create_grouped_boxplot(ax, rel_dev_overcast, rel_dev_clear, color)
        else:
            scene_statistics = [
                sorted_stats[scene][element] for scene in sorted_stats.keys()
            ]
            relative_deviations = [
                np.abs(
                    (scene_statistic - np.mean(scene_statistic))
                    / (np.mean(scene_statistic) + 1e-6)
                )
                * 100
                for scene_statistic in scene_statistics
            ]
            ax.boxplot(
                relative_deviations,
                showfliers=False,
                medianprops={"color": color},
                widths=0.8,
            )
            ax.set_xticks(np.arange(1, len(sorted_stats) + 1))
            ax.set_xticklabels(
                [f"{i}" for i in range(1, len(sorted_stats) + 1)], fontsize=12
            )

        ax.set_title(f"{label} (Deviation from mean)")
        # ax.set_ylim(bottom=0)

    plt.tight_layout(pad=3.0)
    return fig


def show_plots(statistics: dict):
    """Show plots for all parts."""
    # Create and save P1 plots
    fig = create_single_part_plots(statistics, "P1")
    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_p1.png"
    fig.savefig(plot_path, dpi=500)
    print(f"💾 Plot saved at {plot_path}")
    plt.close(fig)

    # Create and save P2 plots
    fig = create_single_part_plots(statistics, "P2")
    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_p2.png"
    fig.savefig(plot_path, dpi=500)
    print(f"💾 Plot saved at {plot_path}")
    plt.close(fig)

    # Create and save P3 plots
    fig = create_single_part_plots(statistics, "P3")
    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_p3.png"
    fig.savefig(plot_path, dpi=500)
    print(f"💾 Plot saved at {plot_path}")
    plt.close(fig)

    # Create and save P2 violin plots
    fig = create_violin_plots(statistics)
    plot_path = f"{GENERATED_PATH}gld_lcim/statistics_p2_violin.png"
    fig.savefig(plot_path, dpi=500)
    print(f"💾 Violin plots saved at {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    statistics = np.load(STATISTICS_PATH, allow_pickle=True).item()
    show_plots(statistics)
