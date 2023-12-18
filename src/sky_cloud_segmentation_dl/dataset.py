import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Tuple, List

import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader

from src.utils.data_utils import UnseededDataLoader, SeededDataLoader

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
SCS_DATA_PATH = DATA_PATH + "sky_cloud_segmentation/"

SKY_FINDER_SEG_TR_PATH = SCS_DATA_PATH + "sky_finder_segmented/train/"
SKY_FINDER_SEG_TR_IMAGES_PATH = SKY_FINDER_SEG_TR_PATH + "images/"
SKY_FINDER_SEG_TR_LABELS_PATH = SKY_FINDER_SEG_TR_PATH + "labels/"

SKY_FINDER_SEG_TE_PATH = SCS_DATA_PATH + "sky_finder_segmented/test/"
SKY_FINDER_SEG_TE_IMAGES_PATH = SKY_FINDER_SEG_TE_PATH + "images/"
SKY_FINDER_SEG_TE_LABELS_PATH = SKY_FINDER_SEG_TE_PATH + "labels/"

N_DUPLICATES = 1000

IMAGE_HEIGHT, IMAGE_WIDTH = 480, 640

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

N_TRAIN_WORKERS, N_VAL_WORKERS = 0, 0

SEED = 42


class SCSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        n_duplicates: int = 1,
        separate_clouds: bool = False,
        deterministic: bool = False,
    ):
        """
        The sky/cloud segmentation dataset.

        Args:
            image_paths (List[str]): The list of images paths
            label_paths (List[str]): The list of labels paths, in the same order as input images
            n_duplicates (int, optional): The number of times to duplicate the dataset, defaults to 1
            separate_clouds (bool, optional): Whether to separate light and thick clouds, defaults to False
            deterministic (bool, optional): Whether to use deterministic transforms, defaults to False

        Returns:
            SCSegmentationDataset: The sky/cloud segmentation dataset

        """
        self.image_paths = image_paths * n_duplicates
        self.label_paths = label_paths * n_duplicates
        self.image_height = IMAGE_HEIGHT
        self.image_width = IMAGE_WIDTH
        self.separate_clouds = separate_clouds
        self.deterministic = deterministic

        # Torch transforms
        self.normalize_torch_transforms = A.Compose(
            [
                A.ToFloat(
                    max_value=1.0
                ),  # Don't need to scale to [0, 1] since normalize does that
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),  # Convert to PyTorch tensor
            ],
            p=1.0,
        )

        self.unnormalize_torch_transforms = A.Compose(
            [
                A.ToFloat(max_value=1.0),  # Ensure image is in [0, 1] range
                ToTensorV2(),  # Convert to PyTorch tensor
            ],
            p=1.0,
        )

        # Common transforms
        deterministic_common_transforms = A.Compose(
            [A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)]
        )

        undeterministic_common_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.HorizontalFlip(p=0.5),
                A.Flip(p=0.5),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                        A.ElasticTransform(p=0.3),
                    ],
                    p=0.8,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=45,
                    value=(0, 0, 0),
                    border_mode=0,
                    p=0.8,
                ),
            ]
        )

        self.common_transforms = (
            deterministic_common_transforms
            if self.deterministic
            else undeterministic_common_transforms
        )

        # Image transforms
        deterministic_image_transforms = A.Compose([])

        undeterministic_image_transforms = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(brightness_limit=(0.0, 0.3), p=1.0),
                        A.RandomGamma(gamma_limit=(25, 100), p=1.0),
                    ],
                    p=1.0,
                ),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.CLAHE(p=1.0),
                    ],
                    p=0.5,
                ),
            ]
        )

        self.image_transforms = (
            deterministic_image_transforms
            if self.deterministic
            else undeterministic_image_transforms
        )

    def __len__(self) -> int:
        """
        Get the dataset length.

        Returns:
            int: The ataset length
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the item at the given index.

        Args:
            idx (int): The index of the item to get

        Returns:
            image (torch.Tensor): The input image, with RGB channels with values in [0, 1]
            label (torch.Tensor): The label image
        """
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Get image and mask
        image = np.array(
            Image.open(image_path).resize((self.image_width, self.image_height))
        )
        label = np.array(
            Image.open(label_path).resize((self.image_width, self.image_height))
        )

        # Get label masks
        r_label = label[:, :, 0]
        g_label = label[:, :, 1]
        b_label = label[:, :, 2]
        ground_mask = np.where(r_label + g_label + b_label < 10, 1, 0).astype(bool)
        sky_mask = np.where(b_label - r_label - g_label > 128, 1, 0).astype(bool)
        light_cloud_mask = np.where(g_label - r_label - b_label > 128, 1, 0).astype(
            bool
        )
        thick_cloud_mask = np.where(r_label - g_label - b_label > 128, 1, 0).astype(
            bool
        )

        # Remove ground pixels from image
        image = image * ~np.repeat(ground_mask[:, :, np.newaxis], 3, axis=2)

        # Get layered label
        if self.separate_clouds:
            label = np.concatenate(
                (
                    ground_mask[:, :, np.newaxis].astype(int),
                    sky_mask[:, :, np.newaxis].astype(int),
                    light_cloud_mask[:, :, np.newaxis].astype(int),
                    thick_cloud_mask[:, :, np.newaxis].astype(int),
                ),
                axis=2,
            )
        else:
            label = np.concatenate(
                (
                    ground_mask[:, :, np.newaxis].astype(int),
                    sky_mask[:, :, np.newaxis].astype(int),
                    (light_cloud_mask | thick_cloud_mask)[:, :, np.newaxis].astype(int),
                ),
                axis=2,
            )

        # Apply common transforms
        data = self.common_transforms(image=image, mask=label)
        image = data["image"]
        label = data["mask"]

        # Empty pixels are labeled as ground
        empty_label = (label[:, :, 0] + label[:, :, 1] + label[:, :, 2] == 0)[
            :, :, np.newaxis
        ]
        label = np.where(
            empty_label, [1, 0, 0, 0] if self.separate_clouds else [1, 0, 0], label
        )

        # Apply image transforms
        image = self.image_transforms(image=image)["image"]

        # Apply torch transforms
        image = self.normalize_torch_transforms(image=image)["image"]
        label = self.unnormalize_torch_transforms(image=label)["image"]

        return image, label


def _get_data_paths(
    images_path: str, 
    labels_path: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Return a list of input and mask paths for the sky finder dataset.

    Args:
        images_path (str): The path to the input images
        labels_path (str): The path to the label images

    Returns:
        image_paths: The list of input paths
        label_paths: The list of label paths
    """

    image_paths = []
    label_paths = []

    # Sky finder segmented
    input_folder_paths = sorted(
        [os.path.join(images_path, folder) for folder in os.listdir(images_path)]
    )
    for folder_path in input_folder_paths:
        folder_image_paths = sorted(
            [
                os.path.join(folder_path, filename)
                for filename in os.listdir(folder_path)
            ]
        )
        for folder_image_path in folder_image_paths:
            image_paths.append(folder_image_path)
            label_paths.append(
                folder_image_path.replace(images_path, labels_path).replace(
                    ".jpg", ".png"
                )
            )

    return image_paths, label_paths


def get_dataloaders(
    batch_size: int, 
    separate_clouds: bool = False, 
    use_workers: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation data loaders.

    Args:
        batch_size (int): The batch size
        separate_clouds (bool, optional): Whether to separate light and thick clouds, defaults to False
        use_workers (bool, optional): Whether to use workers, defaults to True

    Returns:
        train_loader (torch.utils.data.DataLoader): The train data loader
        val_loader (torch.utils.data.DataLoader): The validation data loader
    """

    # Get data paths
    image_paths_tr, label_paths_tr = _get_data_paths(
        SKY_FINDER_SEG_TR_IMAGES_PATH,
        SKY_FINDER_SEG_TR_LABELS_PATH,
    )
    image_paths_val, label_paths_val = _get_data_paths(
        SKY_FINDER_SEG_TE_IMAGES_PATH,
        SKY_FINDER_SEG_TE_LABELS_PATH,
    )

    # Create datasets
    train_dataset = SCSegmentationDataset(
        image_paths_tr,
        label_paths_tr,
        n_duplicates=N_DUPLICATES,
        separate_clouds=separate_clouds,
        deterministic=False,
    )
    val_dataset = SCSegmentationDataset(
        image_paths_val,
        label_paths_val,
        separate_clouds=separate_clouds,
        deterministic=True,
    )

    # Create data loaders
    train_loader = UnseededDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=N_TRAIN_WORKERS if use_workers else 0,
    )
    val_loader = SeededDataLoader(
        SEED,
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=N_VAL_WORKERS if use_workers else 0,
    )

    print("✅ Loaded data.")
    print("➡️ Number of train images:", len(image_paths_tr))
    print("➡️ Number of val images:", len(image_paths_val))

    return train_loader, val_loader
