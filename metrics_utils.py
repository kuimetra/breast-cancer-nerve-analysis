from model import UNet

from torch.utils.data import DataLoader
from typing import Union
import pandas as pd
import numpy as np
import torch


def rotate_mask(mask, angle):
    """
    Rotates the mask by a specified angle.
    """
    return np.rot90(mask, k=angle // 90)


def rotate_back(mask, angle):
    """
    Rotates the mask back to its original orientation after TTA.
    """
    return np.rot90(mask, k=(4 - angle // 90))  # Rotate in opposite direction


def get_tta_predictions(model_file_path: str,
                        data_loader: Union[DataLoader],
                        threshold: float = 0.5,
                        device: str = "cuda") -> tuple[dict, dict]:
    """
    Get model predictions using Test Time Augmentation (TTA).

    Parameters:
    - model_file_path: Path to the model file.
    - data_loader: DataLoader containing the input images.
    - threshold: Threshold for binarizing the model outputs.
    - device: Device to run the model on.

    Returns:
    - inputs_dict: Dictionary mapping case IDs to input images.
    - pred_masks_dict: Dictionary mapping case IDs to predicted masks after TTA.
    """
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_file_path, weights_only=True, map_location=device))
    model.eval()

    inputs_dict, pred_masks_dict = {}, {}

    angles = [0, 90, 180, 270]  # Angles for TTA
    with torch.no_grad():
        for case_ids, inputs, _ in data_loader:
            inputs = inputs.to(device)

            for i, cid in enumerate(case_ids):
                all_preds = []

                for angle in angles:
                    input_img = inputs[i].squeeze().cpu().numpy()
                    rotated_input = rotate_mask(input_img, angle).copy()
                    rotated_input = torch.from_numpy(rotated_input).unsqueeze(0).unsqueeze(0).float().to(device)

                    output = torch.sigmoid(model(rotated_input))

                    # Rotate prediction back to original orientation
                    output = output.squeeze().cpu().numpy()
                    restored_pred = rotate_back(output, angle)

                    all_preds.append(restored_pred)

                # Average predictions and apply threshold
                avg_pred = np.mean(all_preds, axis=0)
                final_mask = (avg_pred > threshold).astype(np.float32)

                inputs_dict[cid] = input_img
                pred_masks_dict[cid] = final_mask

    return inputs_dict, pred_masks_dict


def get_model_predictions(model_file_path: str,
                          data_loader: Union[DataLoader],
                          threshold: float = 0.5,
                          device: str = "cuda") -> tuple[dict, dict]:
    """
    Get model predictions.

    Parameters:
    - model_file_path: Path to the model file.
    - data_loader: DataLoader containing the input images.
    - threshold: Threshold for binarizing the model outputs.
    - device: Device to run the model on.

    Returns:
    - inputs_dict: Dictionary mapping case IDs to input images.
    - pred_masks_dict: Dictionary mapping case IDs to predicted masks.
    """
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_file_path, weights_only=True, map_location=device))
    model.eval()

    inputs_dict, pred_masks_dict = {}, {}

    with torch.no_grad():
        for case_ids, inputs, _ in data_loader:
            inputs = inputs.to(device)

            outputs = torch.sigmoid(model(inputs))
            pred_masks = (outputs > threshold).cpu().numpy().astype(np.float32)

            for i, cid in enumerate(case_ids):
                inputs_dict[cid] = inputs[i].cpu().numpy().squeeze()
                pred_masks_dict[cid] = pred_masks[i].squeeze()

    return inputs_dict, pred_masks_dict


def evaluate_predictions(pred_masks: dict,
                         loose_masks: dict,
                         extended: bool = False) -> Union[dict, tuple]:
    """
    Evaluate the model predictions against loose ground truth masks. Computes Dice Similarity Coefficient (DSC)
    and True Positive Rate (TPR) for each case and returns overall metrics.

    Parameters:
    - pred_masks: Dictionary of predicted masks where keys are case IDs.
    - loose_maks: Dictionary of loose ground truth masks where keys are case IDs.
    - extended: If True, returns metrics per case; otherwise, returns overall metrics.

    Returns:
    - metrics_per_case: Dictionary with case IDs as keys and metrics as values (if extended is True).
    - metrics: Dictionary with overall metrics (DSC and TPR).
    """
    patient_meta_data = pd.read_csv("data/patient_meta.csv")

    metrics_per_case = {}
    dc_scores, tpr_scores = [], []

    for cid in pred_masks.keys():
        loose_mask, pred_mask = loose_masks[cid], pred_masks[cid]

        nerve_count = patient_meta_data.loc[patient_meta_data["tma_case"] == cid, "nerve_count"].values[0]

        if nerve_count != 0:
            # Compute Dice score & TPR only for cases with nerves
            dc = dice_coefficient(loose_mask, pred_mask)
            tpr = true_positive_rate(loose_mask, pred_mask)

            dc_scores.append(dc)
            tpr_scores.append(tpr)

            if extended:
                metrics_per_case[cid] = {"dsc": dc, "tpr": tpr}

    metrics = {"DSC": np.mean(dc_scores), "TPR": np.mean(tpr_scores)}

    return (metrics_per_case, metrics) if extended else metrics


def dice_coefficient(loose_mask, pred_mask, smooth=1e-5):
    """
    Computes the Dice Similarity Coefficient (DSC) between the predicted mask and the loose ground truth mask.
    """
    tp = np.sum((pred_mask == 1) & (loose_mask == 1))
    fp = np.sum((pred_mask == 1) & (loose_mask == 0))
    fn = np.sum((pred_mask == 0) & (loose_mask == 1))
    return (2 * tp) / (2 * tp + fp + fn + smooth)


def true_positive_rate(loose_mask, pred_mask, smooth=1e-5):
    """
    Computes the True Positive Rate (TPR) for the predicted mask against the loose ground truth mask.
    """
    tp = np.sum((pred_mask == 1) & (loose_mask == 1))
    fn = np.sum((pred_mask == 0) & (loose_mask == 1))
    return tp / (tp + fn + smooth)
