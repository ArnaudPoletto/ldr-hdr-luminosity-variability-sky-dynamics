import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import os
import cv2
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.segmentation import slic
import matplotlib.gridspec as gridspec

import torch

from src.utils.video_utils import get_video, get_video_frame_iterator
from src.utils.ground_utils import get_mask as get_ground_mask
from src.utils.ground_utils import get_model_from as get_ground_model_from
from src.utils.cloud_utils import get_mask as get_cloud_mask
from src.utils.cloud_utils import get_model_from as get_cloud_model_from
from src.utils.random_utils import set_seed

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
SCENES_PATH = DATA_PATH + f"ldr/processed/"
GENERATED_PATH = str(GLOBAL_DIR / "generated") + "/"
GLD_LCIM_PATH = DATA_PATH + f"gld_lcim/"
FRAME_STEP = 75

N_SEGMENTS = 1000
COMPACTNESS = 3
SIGMA = 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42


def get_frame_mean(frame, mask=None):
    if mask is not None:
        frame = frame[mask]

    return np.mean(frame)


def get_frame_diff(frame1, frame2, mask=None, abs=False):
    """
    Get the difference between two frames.

    Args:
        frame1 (np.ndarray): The first frame
        frame2 (np.ndarray): The second frame
        mask (np.ndarray, optional): The mask to apply to the frames, defaults to None
        abs (bool, optional): Whether to take the absolute value of the difference, defaults to False

    Returns:
        diff (np.ndarray): The difference between the two frames
    """
    diff = frame1.astype(np.float32) - frame2.astype(np.float32)

    if mask is not None:
        diff = diff[mask]

    if abs:
        np.abs(diff, out=diff)

    return diff


def get_frame_channel_diff_sum(frame1, frame2, channel):
    """
    Get the sum of the difference between two frames for a given channel.

    Args:
        frame1 (np.ndarray): The first frame
        frame2 (np.ndarray): The second frame
        channel (int): The channel to compute the difference for

    Returns:
        diff_sum (float): The sum of the difference between the two frames for the given channel
    """
    return np.sum(
        get_frame_diff(frame1[:, :, channel], frame2[:, :, channel], abs=True)
    )


def get_flow(previous_l_channel, l_channel, mask=None):
    """
    Get the optical flow between two frames.

    Args:
        previous_l_channel (np.ndarray): The previous L channel
        l_channel (np.ndarray): The current L channel
        mask (np.ndarray, optional): The mask to apply to the frames, defaults to None

    Returns:
        flow (np.ndarray): The optical flow between the two frames
    """
    flow = cv2.calcOpticalFlowFarneback(
        previous_l_channel,
        l_channel,
        flow=None,
        pyr_scale=0.1,
        levels=10,
        winsize=31,
        iterations=10,
        poly_n=7,
        poly_sigma=1.2,
        flags=0,
    )

    if mask is not None:
        flow = flow * mask[:, :, np.newaxis]

    return flow


def analyze_scene(
    video_path: str,
    ground_model,
    cloud_model,
    frame_step: int = 1,
    split: bool = False,
    masked: bool = False,
    reframed: bool = False,
):
    """
    Analyze a scene.

    Args:
        video_path (str): The path to the video
        ground_model (torch.nn.Module): The ground segmentation model
        cloud_model (torch.nn.Module): The cloud segmentation model
        frame_step (int, optional): The frame step, defaults to 1
        split (bool, optional): Whether to split the frame in two, defaults to False
        masked (bool, optional): Whether to apply the mask to the frame, defaults to False
        reframed (bool, optional): Whether to reframe the frame, defaults to False

    Returns:
        g_means (list): The list of G means
        diff_g_sums (list): The list of L1 G differences
        cloud_percents (list): The list of cloud percents
        light_percents (list): The list of light percents
        flow_sums (list): The list of flow sums
    """
    video = get_video(video_path)
    frame_it = get_video_frame_iterator(
        video, frame_step=frame_step, split=split, masked=masked, reframed=reframed
    )

    g_means = []
    diff_g_sums = []
    cloud_percents = []
    light_percents = []
    flow_sums = []

    previous_rgb_frame = None
    previous_l_channel = None
    ground_mask = None
    for frame, _ in tqdm(frame_it):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB

        # Reframe shape like previous frame if needed
        if previous_rgb_frame is not None and frame.shape != previous_rgb_frame.shape:
            rgb_frame = cv2.resize(
                rgb_frame, (previous_rgb_frame.shape[1], previous_rgb_frame.shape[0])
            )

        # Get ground mask if needed and apply it to frame
        if ground_mask is None:
            ground_mask = get_ground_mask(rgb_frame / 255, ground_model, factor=0.5)
        rgb_frame = (
            rgb_frame * ground_mask[:, :, np.newaxis]
        )  # Apply ground mask to frame

        # G channel acts as L channel
        l_channel = rgb_frame[:, :, 1]

        # Apply superpixel segmentation
        superpixels = slic(
            rgb_frame,
            n_segments=N_SEGMENTS,
            compactness=COMPACTNESS,
            sigma=SIGMA,
            mask=ground_mask,
        )

        # Replace ground pixels by mean of sky
        masked_l_channel = l_channel.copy()
        masked_l_channel[~ground_mask] = np.mean(l_channel[ground_mask])

        # Rebuild image based on superpixels
        sp_masked_l_channel = masked_l_channel.copy()
        for superpixel_value in np.unique(superpixels):
            mask = superpixels == superpixel_value
            sp_masked_l_channel[mask] = np.median(masked_l_channel[mask])

        # Apply Otsu thresholding and apply mask
        _, sp_lighting_mask = cv2.threshold(
            sp_masked_l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        sp_lighting_mask = sp_lighting_mask.astype(bool)
        sp_lighting_mask = sp_lighting_mask * ground_mask

        # Get statistics
        if previous_rgb_frame is not None:
            # G mean
            g_mean = get_frame_mean(rgb_frame[:, :, 1], mask=ground_mask)

            # L1 G difference
            diff_g_sum = get_frame_channel_diff_sum(rgb_frame, previous_rgb_frame, 1)

            # Cloud percent
            cloud_mask = get_cloud_mask(rgb_frame / 255.0, cloud_model, factor=0.5)
            cloud_mask = (cloud_mask == 2) * ground_mask
            cloud_percent = np.sum(cloud_mask) / np.sum(ground_mask)

            # Light percent
            light_percent = np.sum(sp_lighting_mask) / np.sum(ground_mask)

            # Flow sum
            flow = get_flow(previous_l_channel, l_channel, mask=ground_mask)
            magnitudes = np.linalg.norm(flow, axis=2)
            flow_sum = np.sum(magnitudes)
            flow_sum = flow_sum / np.sum(
                ground_mask
            )  # Normalize by number of sky pixels
            flow_sums.append(flow_sum)

            g_means.append(g_mean)
            diff_g_sums.append(diff_g_sum)
            cloud_percents.append(cloud_percent)
            light_percents.append(light_percent)

            # Show plot with video processing steps
            # plt.figure(figsize=(20, 12))
            # gs = gridspec.GridSpec(3, 2)

            # ax0 = plt.subplot(gs[0, 0])
            # ax0.imshow(rgb_frame)
            # ax0.set_title(f"RGB Frame")

            # ax1 = plt.subplot(gs[1, 0])
            # ax1.imshow(l_channel)
            # ax1.set_title(f"G Channel")

            # ax3 = plt.subplot(gs[1, 1])
            # ax3.imshow(sp_masked_l_channel)
            # ax3.set_title(f"Superpixel G Channel")

            # ax2 = plt.subplot(gs[2, 0])
            # ax2.imshow(cloud_mask, cmap='gray')
            # ax2.set_title(f"Cloud Mask")

            # ax4 = plt.subplot(gs[2, 1])
            # ax4.imshow(sp_lighting_mask, cmap='gray')
            # ax4.set_title(f"Lighting Mask")

            # plt.suptitle(f"Video processing for {video_path.split('/')[-1]} with frame step {frame_step}")

            # plt.tight_layout()
            # plt.show()

        previous_rgb_frame = rgb_frame
        previous_l_channel = l_channel

    # Plot statistics
    plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(6, 2)

    # show first frame and ground mask
    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(frame[:, :, ::-1])  # BGR to RGB
    ax0.set_title(f"Last frame")
    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(ground_mask, cmap="gray")
    ax1.set_title(f"Ground mask")

    ax2 = plt.subplot(gs[1, 0:2])
    ax2.plot(g_means, color="green")
    ax2.set_title(f"G Mean")
    ax2.legend(["G"])

    ax3 = plt.subplot(gs[2, 0:2])
    ax3.plot(diff_g_sums, color="green")
    ax3.set_title(f"L1 G Difference")
    ax3.legend(["G"])

    ax4 = plt.subplot(gs[3, 0:2])
    ax4.plot(cloud_percents, color="red")
    ax4.set_title(f"Cloud percent")
    ax4.legend(["Cloud"])

    ax5 = plt.subplot(gs[4, 0:2])
    ax5.plot(light_percents, color="red")
    ax5.set_title(f"Light percent")
    ax5.legend(["Light"])

    ax6 = plt.subplot(gs[5, 0:2])
    ax6.plot(flow_sums, color="blue")
    ax6.set_title(f"Flow sum")
    ax6.legend(["Flow"])

    plt.suptitle(
        f"Video statistics for {video_path.split('/')[-1]} with frame step {frame_step}"
    )

    plt.tight_layout()

    # Save plot
    generated_gld_lcim_path = f"{GENERATED_PATH}gld_lcim/"
    if not os.path.exists(generated_gld_lcim_path):
        os.makedirs(generated_gld_lcim_path)
    plot_path = (
        generated_gld_lcim_path
        + f"{video_path.split('/')[-1].replace('.mp4', '')}_frame_step_{frame_step}.png"
    )
    plt.savefig(plot_path)
    print(f"💾 Plot saved at {plot_path}")

    # Return list of statistics
    return g_means, diff_g_sums, cloud_percents, light_percents, flow_sums


if __name__ == "__main__":
    set_seed(SEED)

    # Get ground and cloud models
    ground_model_type = "deeplabv3mobilenetv3large"  # "deeplabv3resnet101" is too computationally expensive
    ground_model_save_path = f"{DATA_PATH}sky_ground_segmentation/models/{ground_model_type}_ranger_pretrained.pth"
    ground_model = get_ground_model_from(
        model_save_path=ground_model_save_path, model_type=ground_model_type
    )

    cloud_model_save_path = f"{DATA_PATH}sky_cloud_segmentation/models/deeplabv3resnet101_ranger_pretrained.pth"
    cloud_model = get_cloud_model_from(model_save_path=cloud_model_save_path)

    final_statistics = {}
    scene_paths = [
        os.path.join(SCENES_PATH, scene) for scene in os.listdir(SCENES_PATH)
    ]
    for scene_path in tqdm(scene_paths, desc="▶️  Analyzing scenes"):
        g_means, diff_g_sums, cloud_percents, light_percents, flow_sums = analyze_scene(
            scene_path,
            ground_model,
            cloud_model,
            frame_step=FRAME_STEP,
            split=False,
            masked=True,
            reframed=True,
        )

        final_statistics[scene_path.split("/")[-1]] = {
            "g_means": g_means,
            "diff_g_sums": diff_g_sums,
            "cloud_percents": cloud_percents,
            "light_percents": light_percents,
            "flow_sums": flow_sums,
        }

        # Save statistics
        if not os.path.exists(GLD_LCIM_PATH):
            os.makedirs(GLD_LCIM_PATH)
        statistics_path = f"{GLD_LCIM_PATH}statistics.npy"
        np.save(statistics_path, final_statistics)
        print(f"💾 Statistics saved at {statistics_path}")
