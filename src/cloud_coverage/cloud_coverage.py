import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Tuple, List

import cv2
import Imath
import OpenEXR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from src.utils.ground_utils import get_model_from as get_ground_model_from
from src.utils.ground_utils import get_mask as get_ground_mask
from src.utils.cloud_utils import get_model_from as get_cloud_model_from
from src.utils.cloud_utils import get_mask as get_cloud_mask
from src.utils.random_utils import set_seed
from src.utils.video_utils import get_video, get_frame_from_video

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
HDR_PATH = DATA_PATH + "hdr/"
LDR_PATH = DATA_PATH + "/ldr/processed/"

GENERATED_PATH = str(GLOBAL_DIR / "generated") + "/"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SCENE_MAPPING = {
    1: 8,
    2: 1,
    3: 3,
    4: 6,
    5: 2,
    6: 5,
    7: 4,
    8: 7,
    9: 10,
    10: 9,
}

TIMES = [
    "July 7th, 2023 09:48 - 09:53",
    "August 18th, 2023 15:29 - 15:34",
    "July 6th, 2023 10:53 - 10:58",
    "July 18th, 2023 11:03 - 11:08",
    "August 9th, 2023 10:36 - 10:40",
    "August 16th, 2023 10:22 - 10:26",
    "August 25th, 2023 09:51 - 09:56",
    "July 17th, 2023 10:57 - 11:01",
    "July 24th, 2023 14:20 - 14:25",
    "July 26th, 2023 12:32 - 12:36",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def _get_image_from_exr_file(file_path: str) -> np.ndarray:
    """
    Get the image from an .exr file.

    Args:
        file_path (str): The path to the .exr file

    Returns:
        img (np.ndarray): The HDR RGB image
    """
    file = OpenEXR.InputFile(file_path)

    # Get the image data window and size
    dw = file.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels (assuming RGB format)
    r_str = file.channel("R", Imath.PixelType(Imath.PixelType.FLOAT))
    g_str = file.channel("G", Imath.PixelType(Imath.PixelType.FLOAT))
    b_str = file.channel("B", Imath.PixelType(Imath.PixelType.FLOAT))

    # Convert channel data to numpy arrays
    r = np.frombuffer(r_str, dtype=np.float32)
    g = np.frombuffer(g_str, dtype=np.float32)
    b = np.frombuffer(b_str, dtype=np.float32)

    # Reshape arrays to 2D
    r.shape = (size[1], size[0])
    g.shape = (size[1], size[0])
    b.shape = (size[1], size[0])

    # Stack channels to get RGB image
    img = np.dstack((r, g, b))

    return img


def _reinhard_tone_map(hdr_img: np.ndarray) -> np.ndarray:
    """
    Tone map the HDR image using the Reinhard method.

    Args:
        hdr_img (np.ndarray): The HDR RGB image

    Returns:
        ldr_img (np.ndarray): The associated LDR RGB image with values in [0, 1]
    """
    # Convert HDR to 32-bit floating point
    hdr_img_32f = hdr_img.astype(np.float32)

    if np.any(np.isnan(hdr_img_32f)):
        nan_mask = np.isnan(hdr_img_32f)
        hdr_img_32f[nan_mask] = np.nanmax(hdr_img_32f)

    # Create the tone mapper
    tonemapReinhard = cv2.createTonemapReinhard(1, 0, 0, 0)
    ldr_img = tonemapReinhard.process(hdr_img_32f.copy())

    # Clip and return
    return np.clip(ldr_img, 0, 1)


def _get_clahe_image(ldr_image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the LDR image.

    Args:
        ldr_image (np.ndarray): The LDR RGB image

    Returns:
        ldr_image (np.ndarray): The LDR RGB image with CLAHE applied, with values in [0, 1]
    """
    ldr_image = (ldr_image * 255).astype(np.uint8)
    lab_image = cv2.cvtColor(ldr_image, cv2.COLOR_RGB2LAB)
    l_channel = lab_image[:, :, 0]

    l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    lab_image[:, :, 0] = l_channel

    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB) / 255


def _get_cloud_percentage(
    scene_path: str,
    ground_model: torch.nn.Module,
    cloud_model: torch.nn.Module,
    show: bool = False,
    is_hdr: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Get the cloud percentage of the scene.

    Args:
        scene_path (str): The path to the scene
        ground_model (torch.nn.Module): The ground segmentation model
        cloud_model (torch.nn.Module): The cloud segmentation model
        show (bool, optional): Whether to show the image, defaults to False
        is_hdr (bool, optional): Whether the scene is HDR or not, defaults to True

    Returns:
        cloud_percentage (float): The cloud percentage of the scene
        show_image (np.ndarray): The image with the segmentation colors
    """
    if is_hdr:
        # HDR to LDR or LDR transformation
        image = _get_image_from_exr_file(scene_path)
        ldr_image = _reinhard_tone_map(image)
        ldr_image = np.nan_to_num(ldr_image)
        ldr_image = _get_clahe_image(ldr_image)
    else:
        # Get first frame of LDR video
        video = get_video(scene_path)
        ldr_image, _ = get_frame_from_video(
            video, 0, split=False, masked=True, reframed=True
        )
        ldr_image = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2RGB) / 255
        image = ldr_image.copy()

    # Get ground mask and mask image
    ground_mask = get_ground_mask(ldr_image, ground_model)

    ldr_image_not_masked = ldr_image.copy()
    ldr_image = ldr_image * np.expand_dims(ground_mask, axis=-1)
    image = image * np.expand_dims(ground_mask, axis=-1)

    # Show if needed
    if show:
        plt.imshow(ldr_image)
        plt.title("Original Image")
        plt.show()

    # Get cloud mask
    cloud_mask = get_cloud_mask(ldr_image, cloud_model, factor=0.25 if is_hdr else 0.5)

    # Get cloud percentage
    cloud_percentage = np.sum(cloud_mask == 2) / np.sum(cloud_mask > 0)

    # Get segmentation colors and show if needed
    show_image = ldr_image_not_masked.copy()
    show_image[cloud_mask == 1] = [0, 0, 1]
    show_image[cloud_mask == 2] = [1, 0, 0]
    if show:
        plt.imshow(show_image)
        plt.title(f"Cloud Segmentation ({cloud_percentage:.2%} cloud coverage)")
        plt.show()

    return cloud_percentage, show_image


def _get_cloud_percentage_data(
    path: str,
    file_format: str,
    ground_model: torch.nn.Module,
    cloud_model: torch.nn.Module,
    is_hdr: bool,
) -> dict:
    """
    Get the cloud percentage data for the scenes.

    Args:
        path (str): The path to the scenes
        file_format (str): The file format of the scenes
        ground_model (torch.nn.Module): The ground segmentation model
        cloud_model (torch.nn.Module): The cloud segmentation model
        is_hdr (bool): Whether the scenes are HDR or not

    Returns:
        data (dict): The cloud percentage data for the scenes
    """
    data = {}
    for i in tqdm(range(1, 11), desc="🔄 Processing scenes"):
        scene_num = f"{i:02}" if not is_hdr else str(i)
        scene_path = f"{path}{file_format}{scene_num}.{'exr' if is_hdr else 'mp4'}"
        cloud_percentage, show_image = _get_cloud_percentage(
            scene_path, ground_model, cloud_model, show=False, is_hdr=is_hdr
        )
        data[
            f'{file_format}{scene_num}.{"exr" if is_hdr else "mp4"}'
        ] = {"cloud_percentage": cloud_percentage, "show_image": show_image}

    return data


def _show_cloud_percentage_data(
    data: dict,
    file_format: str,
    times: List[str],
    is_hdr: bool,
    scene_mapping: dict = None,
) -> None:
    """
    Show the cloud percentage data for the scenes.

    Args:
        data (dict): The cloud percentage data for the scenes
        file_format (str): The file format of the scenes
        times (List[str]): The times of the scenes
        is_hdr (bool): Whether the scenes are HDR or not
        scene_mapping (dict, optional): The mapping from scene number to scene number, defaults to None
    """
    figsize = (18, 8) if is_hdr else (25, 6)
    _, axes = plt.subplots(nrows=2, ncols=5, figsize=figsize)
    axes = axes.flatten()
    for i in range(1, 11):
        scene_num = scene_mapping[i] if scene_mapping else i
        idx = scene_num - 1

        cloud_percentage = data[
            f'{file_format}{f"{i:02}" if not is_hdr else str(i)}.{"exr" if is_hdr else "mp4"}'
        ]["cloud_percentage"]
        show_image = data[
            f'{file_format}{f"{i:02}" if not is_hdr else str(i)}.{"exr" if is_hdr else "mp4"}'
        ]["show_image"]

        axes[idx].imshow(show_image)
        axes[idx].axis("off")  # Hide the axis
        axes[idx].set_title(
            f"SCENE {scene_num}\n{cloud_percentage:.1%} clouds", fontsize=18
        )
        axes[idx].text(
            0.5,
            -0.05,
            times[idx],
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[idx].transAxes,
        )

    # Save the figure
    generated_cloud_coverage_path = f"{GENERATED_PATH}cloud_coverage/"
    if not os.path.exists(generated_cloud_coverage_path):
        os.makedirs(generated_cloud_coverage_path)
    plt.savefig(
        f"{generated_cloud_coverage_path}{'hdr' if is_hdr else 'ldr'}_cloud_percentage_figure.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    print(f"💾 Cloud percentage figure saved at {generated_cloud_coverage_path}")

    plt.show()


def _process_scenes(
    ground_model: torch.nn.Module, 
    cloud_model: torch.nn.Module, 
    hdr_path: str, 
    ldr_path: str, 
    times: List[str], 
    scene_mapping: dict = None
):
    """
    Process the scenes.

    Args:
        ground_model (torch.nn.Module): The ground segmentation model
        cloud_model (torch.nn.Module): The cloud segmentation model
        hdr_path (str): The path to the HDR scenes
        ldr_path (str): The path to the LDR scenes
        times (List[str]): The times of the scenes
        scene_mapping (dict, optional): The mapping from scene number to scene number, defaults to None
    """
    choice = input(
        "Enter 'HDR' for HDR processing or 'LDR' for LDR processing: "
    ).upper()
    if choice == "HDR":
        file_format = "scene"
        hdr_data = _get_cloud_percentage_data(
            hdr_path, file_format, ground_model, cloud_model, is_hdr=True
        )
        _show_cloud_percentage_data(hdr_data, file_format, times, True, scene_mapping)
    elif choice == "LDR":
        file_format = "P1Scene"
        ldr_data = _get_cloud_percentage_data(
            ldr_path, file_format, ground_model, cloud_model, is_hdr=False
        )
        _show_cloud_percentage_data(ldr_data, file_format, times, False)
    else:
        raise ValueError(f"❌ Invalid choice: {choice}.")


if __name__ == "__main__":
    set_seed(SEED)

    # Get ground and cloud models
    ground_model_type = "deeplabv3mobilenetv3large" # "deeplabv3resnet101" is too computationally expensive
    ground_model_save_path = f"{DATA_PATH}sky_ground_segmentation/models/{ground_model_type}_ranger_pretrained.pth"
    ground_model = get_ground_model_from(model_save_path=ground_model_save_path, model_type=ground_model_type)
    
    cloud_model_save_path = f"{DATA_PATH}sky_cloud_segmentation/models/deeplabv3resnet101_ranger_pretrained.pth"
    cloud_model = get_cloud_model_from(model_save_path=cloud_model_save_path)

    _process_scenes(ground_model, cloud_model, HDR_PATH, LDR_PATH, TIMES, SCENE_MAPPING)
