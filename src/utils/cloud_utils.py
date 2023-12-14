import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.models.segmentation as models

from src.utils.model_utils import load_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(n_classes: int = 3) -> nn.Module:
    """
    Get the model to be trained.

    Args:
        n_classes (int, optional): The number of classes, defaults to 3

    Returns:
        model (nn.Module): The model to be trained
    """
    model = models.deeplabv3_resnet101(
        weights="COCO_WITH_VOC_LABELS_V1",
        weights_backbone="IMAGENET1K_V2",
        progress=True,
        num_classes=21,
        aux_loss=True,
    ).to(DEVICE)

    # Change the number of classes in the classifier to 2
    in_features = model.classifier[4].in_channels
    new_classifier = nn.Sequential(
        nn.Conv2d(in_features, n_classes, kernel_size=1, stride=1)
    ).to(DEVICE)
    model.classifier[4] = new_classifier

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"✅ Loaded pretrained model deeplabv3resnet101 with {n_params:,} learnable parameters."
    )

    return model


def get_model_from(
    model_save_path: str = "./cloud_model.pth",
) -> torch.nn.Module:
    """
    Get the cloud segmentation model.

    Args:
        model_save_path (str, optional): The path to load the model from, defaults to './cloud_model.pth'
        model_type (str, optional): The model type, defaults to 'deeplabv3resnet101'

    Returns:
        cloud_model (torch.nn.Module): The cloud segmentation model
    """
    cloud_model = get_model(n_classes=3)
    load_model(cloud_model, model_save_path, force=True)

    return cloud_model


def get_mask(
    image: np.ndarray, 
    cloud_model: torch.nn.Module, 
    factor: float = 1.0
) -> np.ndarray:
    """
    Get the cloud mask.

    Args:
        image (np.ndarray): The image
        cloud_model (torch.nn.Module): The cloud segmentation model
        factor (float, optional): The factor by which to resize the image, defaults to 1.0

    Returns:
        cloud_mask (np.ndarray): The cloud mask
    """
    # Image to tensor
    old_image_shape = image.shape
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD  # Normalize
    image = (
        torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    )  # To tensor

    # Get cloud mask
    cloud_model.eval()
    ground_mask = cloud_model(image)["out"].cpu().detach().numpy().squeeze()
    ground_mask = np.argmax(ground_mask, axis=0)

    ground_mask = cv2.resize(
        ground_mask,
        (old_image_shape[1], old_image_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return ground_mask