import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Tuple

import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

from src.utils.video_utils import get_video, get_frame_mask, apply_frame_mask

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
PREPROCESSING_DATA_PATH = DATA_PATH + "ldr/"

RAW_SCENES_PATH = PREPROCESSING_DATA_PATH + "raw/"
PROCESSED_SCENES_PATH = PREPROCESSING_DATA_PATH + "processed/"
PROCESSED_REFRAMED_SCENES_PATH = PREPROCESSING_DATA_PATH + "processed_reframed/"


def _preprocess_frame(
    frame: np.ndarray,
    clahe_clip_limit: float = 3.0,
    clahe_tile_grid_size: Tuple[int, int] = (10, 10),
    bilateral_d: int = 13,
    bilateral_sigma_color: int = 15,
    bilateral_sigma_space: int = 15,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Preprocess the given frame, applying CLAHE for illumination correction and bilateral filtering for noise reduction.

    Args:
        frame (np.ndarray): The frame to be preprocessed
        clahe_clip_limit (float): The clip limit for CLAHE
        clahe_tile_grid_size (Tuple[int, int]): The tile grid size for CLAHE
        bilateral_d (int): The diameter of each pixel neighborhood for bilateral filtering
        bilateral_sigma_color (int): The filter sigma in the color space for bilateral filtering
        bilateral_sigma_space (int): The filter sigma in the coordinate space for bilateral filtering
        mask (np.ndarray): The mask to be applied to the frame

    Returns:
        preprocessed_frame (np.ndarray): The preprocessed frame
    """
    frame = frame.astype(np.uint8)

    # Apply CLAHE for illumination correction
    clahe_frame = frame.copy()
    clahe_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size
    )
    clahe_frame[:, :, 0] = clahe.apply(clahe_frame[:, :, 0])
    clahe_frame = cv2.cvtColor(clahe_frame, cv2.COLOR_LAB2BGR)

    # Only apply CLAHE to masked areas
    if mask is not None:
        frame = np.where(mask[:, :, np.newaxis] == 1, clahe_frame, frame)

    # Apply Bilateral Filtering for noise reduction on the entire frame
    frame = cv2.bilateralFilter(
        frame,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    return frame.astype(np.uint8)


def _mask_reframe_frame(frame: np.ndarray, type: str) -> np.ndarray:
    """
    Apply the left or right mask to the given frame and reframes it.

    Args:
        frame (np.ndarray): The frame to be masked and reframed
        type (str): The type of mask to be applied

    Raises:
        ValueError: If the mask type is invalid

    Returns:
        masked_reframed_frame (np.ndarray): The masked and reframed frame
    """
    if type not in ["left", "right"]:
        raise ValueError(f"❌ Invalid mask type {type}: must be 'left' or 'right'.")

    # Mask
    masked_frame = apply_frame_mask(frame, type)

    # Reframe
    masked_reframed_frame = masked_frame[~np.all(masked_frame == 0, axis=2).all(axis=1)]
    masked_reframed_frame = masked_reframed_frame[
        :, ~np.all(masked_reframed_frame == 0, axis=2).all(axis=0)
    ]

    return masked_reframed_frame


def _process_frames(
    output_video_path: str,
    input_video_path: str,
    start_frame: int,
    end_frame: int,
    scene_file: str,
    mask_reframe: bool = False,
    n_frames_to_average: int = 3,
) -> None:
    """
    Process frames of the given video, writing the processed frames to the output video at the given path.
    Frames are processed by applying CLAHE for illumination correction, bilateral filtering for noise reduction
    and temporal averaging for flicker reduction.

    Args:
        output_video_path (str): The path to the output video
        input_video_path (str): The path to the input video
        start_frame (int): The start frame of the video
        end_frame (int): The end frame of the video
        scene_file (str): The name of the scene file
        mask_reframe (bool): Whether to mask and reframe the frames
        n_frames_to_average (int): The number of frames to average for temporal averaging
    """
    input_video = get_video(input_video_path)
    input_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set start frame

    output_video = None
    frame_buffer = []  # Buffer for averaging frames with temporal averaging
    i = start_frame

    if start_frame == 0:
        bar = tqdm(total=end_frame - start_frame, desc=f"▶️  Processing {scene_file}")
    while i < end_frame:
        ret, frame = input_video.read()
        if not ret:
            break

        # Process left frame
        left_frame, _ = frame[:, : frame.shape[1] // 2], frame[:, frame.shape[1] // 2 :]
        if mask_reframe:
            left_frame = _mask_reframe_frame(left_frame, "left")
        mask = get_frame_mask("left", reframed=mask_reframe)
        preprocessed_frame = _preprocess_frame(left_frame, mask=mask)

        # Apply temporal averaging
        frame_buffer.append(preprocessed_frame)
        preprocessed_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)
        if len(frame_buffer) == n_frames_to_average:
            frame_buffer.pop(0)  # Remove oldest frame from buffer

        # Make sure that the output video is initialized before writing frames
        if output_video is None:
            frame_width = preprocessed_frame.shape[1]
            frame_height = preprocessed_frame.shape[0]
            frame_rate = int(input_video.get(5))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video = cv2.VideoWriter(
                os.path.join(
                    output_video_path,
                    f"{str(start_frame)}-{str(end_frame)}_{scene_file}",
                ),
                fourcc,
                frame_rate,
                (frame_width, frame_height),
            )

        # Write frame to output video
        output_video.write(preprocessed_frame)
        i += 1
        if start_frame == 0:
            bar.update(1)

    input_video.release()


def _process_video(
    output_video_path: str,
    scene_file: str,
    num_processes: int = 4,
    mask_reframe: bool = False,
):
    """
    Process the given video, writing the processed frames to the output video at the given path.

    Args:
        output_video_path (str): The path to the output video
        scene_file (str): The name of the scene file
        num_processes (int): The number of processes to use
        mask_reframe (bool): Whether to mask and reframe the frames
    """
    print(f"▶️  Processing {output_video_path}{scene_file}...")

    # Get number of frames in video
    input_video_path = os.path.join(RAW_SCENES_PATH, scene_file)
    input_video = get_video(input_video_path)
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    input_video.release()

    # Create separate processes for processing frames
    frames_per_process = frame_count // num_processes
    read_processes = []
    for i in range(num_processes):
        start_frame = i * frames_per_process
        end_frame = (
            start_frame + frames_per_process if i < num_processes - 1 else frame_count
        )
        read_process = Process(
            target=_process_frames,
            args=(
                output_video_path,
                input_video_path,
                start_frame,
                end_frame,
                scene_file,
                mask_reframe,
            ),
        )
        read_processes.append(read_process)
        read_process.start()

    # Wait for all processes to finish
    for read_process in read_processes:
        read_process.join()

    # Get all processed videos
    processed_videos = os.listdir(output_video_path)
    processed_videos = [
        video for video in processed_videos if video.endswith(scene_file)
    ]

    # Sort processed videos by start frame
    processed_videos.sort(key=lambda video: int(video.split("_")[0].split("-")[0]))

    # Combine processed videos
    print(f"▶️  Combining {len(processed_videos)} processed videos...")
    output_video = None
    for processed_video in tqdm(processed_videos):
        processed_video_path = os.path.join(output_video_path, processed_video)
        processed_video = get_video(processed_video_path)

        while True:
            ret, frame = processed_video.read()
            if not ret:
                break

            # Make sure that the output video is initialized before writing frames
            if output_video is None:
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]
                frame_rate = int(processed_video.get(5))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_video = cv2.VideoWriter(
                    os.path.join(output_video_path, scene_file),
                    fourcc,
                    frame_rate,
                    (frame_width, frame_height),
                )

            # Write frame to output video
            output_video.write(frame)

        processed_video.release()

    output_video.release()

    # Delete processed videos
    for processed_video in processed_videos:
        os.remove(os.path.join(output_video_path, processed_video))

    print(f"✅ Processed {output_video_path}{scene_file}.")


if __name__ == "__main__":
    # Get number of processes and whether to mask and reframe frames
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_processes",
        type=int,
        default=os.cpu_count(),
        help="The number of processes to use",
    )
    parser.add_argument(
        "--mask_reframe",
        action="store_true",
        help="Whether to mask and reframe the frames",
    )
    args = parser.parse_args()
    num_processes = args.num_processes
    mask_reframe = args.mask_reframe

    print(
        f"▶️  Processing {'and reframing ' if mask_reframe else ''}scenes using {num_processes} processes..."
    )

    # Create output video directory
    output_video_path = (
        PROCESSED_REFRAMED_SCENES_PATH if mask_reframe else PROCESSED_SCENES_PATH
    )
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)

    # Process videos
    if not os.path.exists(RAW_SCENES_PATH):
        print(f"❌ Invalid raw scenes path {RAW_SCENES_PATH}: directory does not exist.")
        exit()

    scene_files = os.listdir(RAW_SCENES_PATH)
    for scene_file in scene_files:
        _process_video(
            output_video_path,
            scene_file,
            num_processes=num_processes,
            mask_reframe=mask_reframe,
        )
