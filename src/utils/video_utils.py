import os
import cv2
import numpy as np
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / '..' / '..'
DATA_PATH = str(GLOBAL_DIR / 'data') + '/'

LEFT_MASK_PATH = DATA_PATH + 'left_mask.png'
RIGHT_MASK_PATH = DATA_PATH + 'right_mask.png'

def get_video(video_path: str) -> cv2.VideoCapture:
    """
    Given the path to a video, return a VideoCapture object for the video.

    Args:
        video_path (str): The path to the video

    Returns:
        cv2.VideoCapture: The VideoCapture object for the video
    """
    if not os.path.exists(video_path):
        raise ValueError(f"❌ Invalid video path {video_path}: file does not exist.")

    return cv2.VideoCapture(video_path)

def get_frame_mask(type: str, reframed: bool = False) -> np.ndarray:
    """
    Return the left or right video mask, optionally reframed.

    Args:
        type (str): The type of mask to be returned
        reframed (bool): Whether to reframe the mask, defaults to False

    Returns:
        mask (np.ndarray): The left or right mask, with values in [0, 1]
    """
    if type not in ['left', 'right']:
        raise ValueError(f"❌ Invalid mask type {type}: must be 'left' or 'right'.")
    
    mask = cv2.imread(LEFT_MASK_PATH if type == 'left' else RIGHT_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask > 255 * 0.5, 1, 0)

    if reframed:
        # Remove rows containing all zeros
        mask = mask[~np.all(mask == 0, axis=1)]

        # Remove columns containing all zeros
        mask = mask[:, ~np.all(mask == 0, axis=0)]

    return mask.astype(np.uint8)

def apply_frame_mask(frame: np.ndarray, type: str) -> np.ndarray:
    """
    Applies the left or right mask to the given frame.

    Args:
        frame (np.ndarray): The frame to be masked
        type (str): The mask type to be applied

    Returns:
        masked_frame (np.ndarray): Masked frame.
    """
    if type not in ['left', 'right']:
        raise ValueError(f"❌ Invalid mask type {type}: must be 'left' or 'right'.")

    mask = get_frame_mask(type)
    masked_frame = frame * mask[:, :, np.newaxis]

    return masked_frame

def get_frame_from_video(video: cv2.VideoCapture, frame: int, split: bool = True, masked: bool = False, reframed: bool = False) -> np.ndarray:
    """
    Returns left and right frames at the given index from the video, optionally masked and reframed.

    Args:
        video (cv2.VideoCapture): The VideoCapture object for the video
        frame (int): The index of the frame to be returned
        split (bool): Whether to return the left and right frames separately, defaults to True
        masked (bool): Whether to return the masked version of the frame, defaults to False
        reframed (bool): Whether to reframe the frame, defaults to False

    Returns:
        left_frame (np.ndarray): Left frame at the given index, with BGR channels in [0, 255].
        right_frame (np.ndarray): Right frame at the given index, with BGR channels in [0, 255].
    """
    if frame < 0 or frame >= video.get(cv2.CAP_PROP_FRAME_COUNT):
        raise ValueError(f"❌ Invalid frame index {frame}: video has {int(video.get(cv2.CAP_PROP_FRAME_COUNT))} frames.")

    video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _, frame = video.read()

    if split:
        left_frame = frame[:, :frame.shape[1]//2]
        right_frame = frame[:, frame.shape[1]//2:]
    else:
        left_frame = frame
        right_frame = frame

    if masked:
        left_frame = apply_frame_mask(left_frame, 'left')
        right_frame = apply_frame_mask(right_frame, 'right')

    if reframed:
        # Remove rows containing all zeros
        left_frame = left_frame[~np.all(left_frame == 0, axis=2).all(axis=1)]
        right_frame = right_frame[~np.all(right_frame == 0, axis=2).all(axis=1)]

        # Remove columns containing all zeros
        left_frame = left_frame[:, ~np.all(left_frame == 0, axis=2).all(axis=0)]
        right_frame = right_frame[:, ~np.all(right_frame == 0, axis=2).all(axis=0)]

    return left_frame.astype(np.uint8), right_frame.astype(np.uint8)

def get_video_frame_iterator(video: cv2.VideoCapture, frame_step: int = 1, split: bool = True, masked: bool = False, reframed: bool = False) -> np.ndarray:
    """
    Return an iterator over the frames of the video, optionally masked and reframed.

    Args:
        video (cv2.VideoCapture): The VideoCapture object for the video
        frame_step (int): The number of frames to skip between each frame returned, defaults to 1
        split (bool): Whether to return the left and right frames separately, defaults to True
        masked (bool): Whether to return the masked version of the frame, defaults to False
        reframe (bool): Whether to reframe the frame, defaults to False

    Returns:
        frame_iterator (np.ndarray): Iterator over the frames of the video, with BGR channels in [0, 255].
    """
    if frame_step < 1:
        raise ValueError(f"❌ Invalid frame step {frame_step}: must be greater than 0.")

    frame = 0
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame < n_frames:
        yield get_frame_from_video(video, frame, split, masked, reframed)

        frame += frame_step