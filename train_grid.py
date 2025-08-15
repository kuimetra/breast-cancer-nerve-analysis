from dataloader_utils import get_dataloaders
from data_io import save_config
import transforms as tfms
from model import UNet
import losses as lss

from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import random
import torch
import os


### ------------------ 1. ARGUMENT PARSING ------------------ ###
def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple U-Net models with different configurations.")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use (default: 0)")
    return parser.parse_args()


### ------------------ 2. SETUP TRAINING ENVIRONMENT ------------------ ###

def setup_environment(experiment_name, learning_rate):
    torch.manual_seed(256)
    np.random.seed(256)
    random.seed(256)

    lr = f"lr{int(-np.log10(learning_rate))}"
    output_folder = os.path.join("models", f"{experiment_name}_{lr}")
    os.makedirs(output_folder, exist_ok=True)

    return output_folder


### ------------------ 3. TRAINING FUNCTION ------------------ ###
def train_network(config):
    """
    Trains a U-Net model and logs all training details.

    Parameters:
    - config: Dictionary containing all training configurations.
    """
    device = config["device"]
    num_epochs = config["num_epochs"]
    patience = config["patience"]
    output_folder = config["output_folder"]
    data_augmentation = config["data_augmentation"]
    transformations = config["transformations"]
    loss_func = config["loss_function"]
    learning_rate = config["learning_rate"]

    loaders = get_dataloaders(mode="train", data_augmentation=data_augmentation, transform_list=transformations)
    train_loader, val_loader = loaders["train"], loaders["val"]

    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = loss_func

    # Learning rate scheduler (reduces lr when validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    train_losses, val_losses, timestamps, learning_rates = [], [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for _, inputs, target_masks in train_loader:
            inputs, target_masks = inputs.to(device), target_masks.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, target_masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, inputs, target_masks in val_loader:
                inputs, target_masks = inputs.to(device), target_masks.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, target_masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        timestamps.append(datetime.now().strftime('%d-%m-%Y %H:%M:%S'))

        scheduler.step(avg_val_loss)  # Adjusts LR when validation loss stops improving

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_weights = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            early_stopping_epoch = epoch + 1
            print(f"Early stopping triggered at epoch {early_stopping_epoch}.")
            config["early_stopping_epoch"] = early_stopping_epoch
            break

        print(
            f"{timestamps[-1]} | Epoch {epoch + 1} | Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | Learning Rate: {current_lr:.5f}"
        )

    torch.save(best_model_weights, os.path.join(output_folder, "model.pth"))
    print(f"Model saved at: {output_folder}/model.pth")

    pd.DataFrame({
        "Timestamp": timestamps,
        "Epoch": range(1, len(train_losses) + 1),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "Learning Rate": learning_rates,
    }).to_csv(os.path.join(output_folder, "losses.csv"), index=False)
    print(f"Loss log saved at: {output_folder}/losses.csv")

    save_config(config, output_folder)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    transformation_sets = {
        "": [],
        "mmn": [tfms.MinMaxNormalize()],
        "log": [tfms.LogTransform()],
        "pn": [tfms.PercentileNormalize()],
        "ci": [tfms.ClipIntensity()],
        "ci_mmn": [tfms.ClipIntensity(), tfms.MinMaxNormalize()],
        "gb_he_mmn": [tfms.GaussianBlur(), tfms.HistogramEqualization(), tfms.MinMaxNormalize()],
    }

    loss_fn_map = {
        "wce": lss.WeightedCrossEntropyLoss,
        "focal": lss.FocalLoss,
        "dice": lss.DiceLoss,
    }

    learning_rates = [1e-3, 1e-4, 1e-5]

    # Grid search
    for loss_key, loss_func in loss_fn_map.items():
        for tfm_key, tfm_list in transformation_sets.items():
            for lr in learning_rates:
                tfm_prefix = f"{tfm_key}_" if tfm_key else ""
                output_name = f"{tfm_prefix}{loss_key}"

                print(f"\n===== Starting training: {output_name} =====")
                output_folder = setup_environment(output_name, lr)

                training_config = {
                    "device": device,
                    "output_folder": output_folder,
                    "num_epochs": 500,
                    "patience": 20,
                    "learning_rate": lr,
                    "loss_function": loss_func(),
                    "data_augmentation": True,
                    "transformations": tfm_list,
                }

                train_network(training_config)
