import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
    """
    Validate the model on the validation set and return the validation loss.

    Args:
        model (nn.Module): The model
        val_loader (DataLoader): The data loader used for validation
        criterion (nn.Module): The criterion

    Returns:
        val_loss (float): The validation loss
    """

    model.eval()

    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            input, mask = data

            # Move to GPU
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)

            output = model(input)
            output = output["out"]
            loss = criterion(output, mask)
            val_loss += loss.item()

    model.train()

    return val_loss / len(val_loader)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epochs: int,
    accumulation_steps: int,
    validation_steps: int,
    info_file_path: str,
    model_save_path: str,
) -> None:
    """
    Train the model for the specified number of epochs on the training set and validate on the validation set. Save the model with the best validation loss to the specified path and save the training and validation loss to the specified info file.

    Args:
        model (nn.Module): The model
        train_loader (DataLoader): The data loader used for training
        val_loader (DataLoader): The data loader used for validation
        criterion (nn.Module): The criterion
        optimizer (optim.Optimizer): The optimizer
        scheduler (optim.lr_scheduler._LRScheduler): The scheduler
        epochs (int): The number of epochs to train for
        accumulation_steps (int): The number of steps to accumulate gradients for
        validation_steps (int): The number of steps to validate for
        info_file_path (str): The path to the info file
        model_save_path (str): The path to save the model to
    """

    print(f"🚀 Training model for {epochs} epochs...")

    best_val_loss = float("inf")
    train_loss = 0

    # Refresh info file
    if os.path.exists(info_file_path):
        open(info_file_path, "w").close()

    with open(info_file_path, "a") as f:
        f.write(f"epoch,step,validation_loss,train_loss\n")

    scaler = GradScaler()

    model.train()

    for epoch in range(epochs):
        bar = tqdm(total=len(train_loader))
        for i, data in enumerate(train_loader):
            inputs, masks = data

            # Move to GPU
            inputs = inputs.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            with autocast():
                try:
                    outputs = model(inputs)
                    outputs = outputs["out"]
                except Exception as e:
                    print(f"❌ {e}")
                    continue
                loss = criterion(outputs, masks)

            # Backward pass
            scaler.scale(loss).backward()
            train_loss += loss.item()

            # Update weights every ACCUMULATION_STEPS
            if (i + 1) % accumulation_steps == 0:
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Decay learning rate
                scheduler.step()

                # Show training loss
                bar.set_description(
                    f"➡️ Training loss: {train_loss / accumulation_steps:.5f}"
                )
                train_loss = 0

            # Validate every VALIDATION_STEPS
            if (i + 1) % validation_steps == 0 or i == 0:
                # Validate
                val_loss = validate(model, val_loader, criterion)

                # Print validation info to file
                with open(info_file_path, "a") as f:
                    f.write(f"{epoch},{i+1},{val_loss},{loss.item()}\n")

                # Save model for best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)
                    bar.clear()
                    print(f"✅ Saved model with validation loss {val_loss:.5f}.")

            bar.update(1)


def load_model(model: nn.Module, model_save_path: str, force: bool = False) -> None:
    """
    Load model from path.

    Args:
        model (nn.Module): The model
        model_save_path (str): The path to load the model from
        force (bool): Whether to force loading the model
    """

    if os.path.exists(model_save_path) or force:
        model.load_state_dict(torch.load(model_save_path))
        print(f"✅ Loaded model from {model_save_path}")
    else:
        print("❌ Model not loaded.")
