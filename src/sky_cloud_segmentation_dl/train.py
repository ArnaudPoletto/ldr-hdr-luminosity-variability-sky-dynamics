import sys
from pathlib import Path
GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ranger_adabelief import RangerAdaBelief

import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler

from src.sky_cloud_segmentation_dl.dataset import get_dataloaders
from src.utils.cloud_utils import get_model
from src.utils.model_utils import train
from src.utils.random_utils import set_seed

DATA_PATH = str(GLOBAL_DIR / "data") + "/"
SCS_DATA_PATH = DATA_PATH + "sky_cloud_segmentation/"

EPOCHS = 2
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
VALIDATION_STEPS = 200
LEARNING_RATE = 0.001
EPSILON = 1e-12
BETAS = (0.95, 0.999)
WEIGHT_DECAY = 0.01
STEP_SIZE = 50
GAMMA = 0.97

CLASS_WEIGHTS = torch.tensor([[2.14396795, 4.16733798, 3.40583472]])
CLASS_WEIGHTS = CLASS_WEIGHTS.view(1, 3, 1, 1)

SEPARATE_CLOUDS = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 27


def get_criterion(class_weights: torch.Tensor = CLASS_WEIGHTS) -> nn.Module:
    """
    Get the criterion to be used for training.

    Args:
        class_weights (torch.Tensor, optional): The class weights, defaults to CLASS_WEIGHTS

    Returns:
        criterion (nn.Module): The criterion to be used for training
    """
    criterion = nn.BCEWithLogitsLoss(weight=class_weights.to(DEVICE))
    print(f"✅ Loaded weighted binary cross entropy loss.")

    return criterion


def get_optimizer(
    model: nn.Module,
    learning_rate: float = LEARNING_RATE,
    epsilon: float = EPSILON,
    betas: tuple = BETAS,
    weight_decay: float = WEIGHT_DECAY,
) -> optim.Optimizer:
    """
    Get the optimizer to be used for training.

    Args:
        model (nn.Module): The modelSCSegmentationDataset
        learning_rate (float, optional): The learning rate, defaults to LEARNING_RATE
        epsilon (float, optional): The optimizer epsilon, defaults to EPSILON
        betas (tuple, optional): The optimizer betas, defaults to BETAS
        weight_decay (float, optional): The weight decay, defaults to WEIGHT_DECAY

    Returns:
        optimizer (optim.Optimizer): The optimizer to be used for training
    """
    optimizer = RangerAdaBelief(
        model.parameters(),
        lr=learning_rate,
        eps=epsilon,
        betas=betas,
        weight_decay=weight_decay,
    )
    print(
        f"✅ Loaded ranger optimizer with learning rate {learning_rate} and weight decay {weight_decay}."
    )

    return optimizer


def get_scheduler(
    optimizer: optim.Optimizer, step_size: int = STEP_SIZE, gamma: float = GAMMA
) -> lr_scheduler._LRScheduler:
    """
    Get the scheduler to be used for training.

    Args:
        optimizer (optim.Optimizer): The optimizer
        step_size (int, optional): The step size, defaults to STEP_SIZE
        gamma (float, optional): The learning rate decay, defaults to GAMMA

    Returns:
        scheduler (lr_scheduler._LRScheduler): The scheduler to be used for training
    """
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    print(f"✅ Loaded scheduler with step size {step_size} and gamma {gamma}.")

    return scheduler

if __name__ == "__main__":
    # Set seed for deterministic results
    set_seed(SEED)

    train_loader, val_loader = get_dataloaders(
        BATCH_SIZE, separate_clouds=SEPARATE_CLOUDS
    )

    model = get_model()

    optimizer = get_optimizer(model)
    criterion = get_criterion()
    scheduler = get_scheduler(optimizer)

    info_file_path = str(
        GLOBAL_DIR
        / "data"
        / "sky_cloud_segmentation"
        / "models"
        / f"deeplabv3resnet101_ranger_pretrained_info.csv"
    )
    model_save_path = str(
        GLOBAL_DIR
        / "data"
        / "sky_cloud_segmentation"
        / "models"
        / f"deeplabv3resnet101_ranger_pretrained.pth"
    )
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        accumulation_steps=ACCUMULATION_STEPS,
        validation_steps=VALIDATION_STEPS,
        info_file_path=info_file_path,
        model_save_path=model_save_path,
    )
