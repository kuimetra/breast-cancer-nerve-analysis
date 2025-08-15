from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Optional
import pandas as pd
import numpy as np
import os

cmu_font = font_manager.FontProperties(fname="cmu.serif-roman.ttf")


def plot_protein_markers(protein_markers: np.ndarray,
                         marker_panel_df: pd.DataFrame,
                         row_start: Optional[int] = None,
                         row_end: Optional[int] = None,
                         col_start: Optional[int] = None,
                         col_end: Optional[int] = None):
    """
    Plots protein marker images.

    Parameters:
    - protein_markers: Array of 38 protein marker images.
    - marker_panel_df: DataFrame containing 'Metal' and 'Category' columns of the protein markers.
    - row_start: Starting row for cropping the image.
    - row_end: Ending row for cropping the image.
    - col_start: Starting column for cropping the image.
    - col_end: Ending column for cropping the image.
    """
    protein_markers = protein_markers[:, row_start:row_end, col_start:col_end]

    metals = marker_panel_df['Metal'].tolist()
    categories = marker_panel_df['Category'].tolist()

    unique_categories = np.unique(categories)
    cmap = plt.get_cmap('tab20b_r')
    custom_cmap = LinearSegmentedColormap.from_list('white_to_red', ['#ffffff', '#DB5461'])
    colors = cmap(np.linspace(0, 1, len(unique_categories)))
    category_to_color = {category: colors[i] for i, category in enumerate(unique_categories)}

    fig, axes = plt.subplots(7, 6, figsize=(15, 20))
    axes = axes.ravel()

    ax_indices = list(range(36)) + [38, 39]

    for i, ax_idx in zip(list(range(38)), ax_indices):
        image_array = protein_markers[i]

        normalized_image = np.log1p(image_array)
        axes[ax_idx].imshow(normalized_image, cmap=custom_cmap)

        current_category = categories[i]
        label_color = category_to_color[current_category]
        rect = plt.Rectangle((0, 1.01), 1, 0.12, transform=axes[ax_idx].transAxes, color=label_color, clip_on=False)
        axes[ax_idx].add_patch(rect)
        cmu_font = font_manager.FontProperties(fname="cmu.serif-roman.ttf")

        axes[ax_idx].text(0.5, 1.06, metals[i], fontsize=16, ha='center', va='center',
                          transform=axes[ax_idx].transAxes, color='white', fontproperties=cmu_font)

        axes[ax_idx].axis('off')

    for idx in [36, 37, 40, 41]:
        axes[idx].axis('off')

    legend_patches = [Patch(color=category_to_color[category], label=category) for category in unique_categories]
    cmu_font = font_manager.FontProperties(fname="cmu.serif-roman.ttf", size=18.9)
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(unique_categories) // 2, fontsize=18.5,
               bbox_to_anchor=(0.5, -0.04), prop=cmu_font)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


def plot_segmented_masks(cell_masks: np.ndarray, cell_meta_df: pd.DataFrame):
    """
    Visualizes cell masks.

    Parameters:
    - cell_masks: 10-layer cell mask array.
    - cell_meta_df: DataFrame containing metadata for the cells, including 'cell_type'.
    """
    cmap_cell = plt.get_cmap('tab10')
    color_array_rgb = np.zeros((*cell_masks[0].shape, 3), dtype=np.float64)

    object_to_color = {}

    for i, cell_type in enumerate(sorted(cell_meta_df['cell_type'].unique())):
        mask = cell_masks[i] != 0
        color = cmap_cell(i)[:3]
        color_array_rgb[mask] = color

        object_to_color[cell_type] = color

    unique_cell_types = sorted(cell_meta_df['cell_type'].unique())

    legend_patches = []
    for cell_type in unique_cell_types:
        color = object_to_color[cell_type]
        patch = Patch(facecolor=color, label=cell_type.replace("_", " "))
        legend_patches.append(patch)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    ax.imshow(color_array_rgb)

    ax.axis('off')

    plt.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.2),
               fontsize=28, ncol=3, frameon=True, prop=cmu_font)

    plt.tight_layout()
    plt.show()


def plot_loss_curves(lr: str, rows: list, columns: list, row_labels: list, col_labels: list):
    """
    Plots loss curves for multiple experiments based on the provided learning rate,
    data preprocessing methods, and loss functions.

    Parameters:
    - lr: Learning rate to filter experiments.
    - rows: Data preprocessing methods to be used as rows in the plot.
    - columns: Loss functions to be used as columns in the plot.
    - row_labels: Labels for the rows in the plot.
    - col_labels: Labels for the columns in the plot.
    """
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(12, 16), dpi=300)

    for row_idx, row_name in enumerate(rows):
        for col_idx, col_name in enumerate(columns):
            if row_name == "-":
                experiment_name = col_name
            else:
                experiment_name = f"{row_name}_{col_name}"

            losses_path = os.path.join("models", f"{experiment_name}_lr{lr}", "losses.csv")

            if os.path.exists(losses_path):
                loss_df = pd.read_csv(losses_path)

                lr_norm = (loss_df["Learning Rate"] - loss_df["Learning Rate"].min()) / (
                        loss_df["Learning Rate"].max() - loss_df["Learning Rate"].min())

                cmap = plt.get_cmap("Spectral")

                ax = axes[row_idx, col_idx]

                for i in range(len(loss_df["Epoch"]) - 1):
                    ax.plot(loss_df["Epoch"][i:i + 2], loss_df["Train Loss"][i:i + 2],
                            color=cmap(lr_norm[i]), linewidth=2)

                ax.plot(loss_df["Epoch"], loss_df["Validation Loss"], label="Val", color="black", linestyle="--",
                        linewidth=1)

                ax.grid(True, linestyle="--", linewidth=0.5)

                cmu_font = font_manager.FontProperties(fname="cmu.serif-roman.ttf", size=10)

                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontproperties(cmu_font)
                    label.set_fontsize(10)

            if row_idx > 0:
                ax.set_title("")
            if col_idx > 0:
                ax.set_title("")

    cmu_font = font_manager.FontProperties(fname="cmu.serif-roman.ttf", size=14)

    for row_idx in range(len(rows)):
        ax = axes[row_idx, 0]
        ax.text(-0.2, 0.5, row_labels[row_idx], va='center', ha='center', rotation=90, fontsize=14,
                transform=ax.transAxes, fontproperties=cmu_font)

    cmu_font = font_manager.FontProperties(fname="cmu.serif-roman.ttf", size=16)

    for col_idx, col_name in enumerate(col_labels):
        ax = axes[0, col_idx]
        ax.set_title(col_name, fontsize=16, va='center', ha='center', y=1.05, fontproperties=cmu_font)

    fig.text(0.53, 1.001, "Loss Function", fontsize=20, ha="center", fontproperties=cmu_font)
    fig.text(-0.015, 0.5, "Data Preprocessing", fontsize=20, va="center", rotation=90, fontproperties=cmu_font)

    plt.tight_layout()
    plt.show()


def plot_loss_curve(model_name: str):
    """
    Plots the training and validation loss curves for a given model.

    Parameters:
    - model_name: The name of the model file containing the loss data.
    """
    loss_df = pd.read_csv(model_name)

    lr_norm = (loss_df["Learning Rate"] - loss_df["Learning Rate"].min()) / (
            loss_df["Learning Rate"].max() - loss_df["Learning Rate"].min())

    cmap = plt.get_cmap("Spectral")

    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

    # Training loss with color changing based on learning rate
    for i in range(len(loss_df["Epoch"]) - 1):
        ax.plot(loss_df["Epoch"][i:i + 2], loss_df["Train Loss"][i:i + 2],
                color=cmap(lr_norm[i]), linewidth=2)

    ax.plot(loss_df["Epoch"], loss_df["Validation Loss"], label="Validation Loss", color="black", linestyle="--",
            linewidth=1)

    cmu_font = font_manager.FontProperties(fname="cmu.serif-roman.ttf", size=14)

    ax.set_xlabel("Epoch", fontsize=12, fontproperties=cmu_font)
    ax.set_ylabel("Loss", fontsize=12, fontproperties=cmu_font)
    ax.set_title(f"Training and Validation Loss", fontsize=16, fontproperties=cmu_font)
    ax.grid(True, linestyle="--", linewidth=0.5)

    gradient_line = Line2D([0], [0], color=cmap(0), linewidth=2, label="Training Loss")

    cmu_font = font_manager.FontProperties(fname="cmu.serif-roman.ttf", size=10)

    ax.legend(handles=[gradient_line, ax.lines[-1]], loc='upper right', fontsize=8, prop=cmu_font)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(cmu_font)
        label.set_fontsize(10)

    plt.tight_layout()
    plt.show()


def peripherin_overlay_comparison(input_image: np.ndarray, loose_mask: np.ndarray, pred_mask: np.ndarray):
    """
    Plots a side-by-side comparison of the peripherin image and the overlay highlighting TP, FP, and FN.

    Parameters:
    - input_image: Peripherin image.
    - loose_mask: Loose ground truth mask for comparison.
    - pred_mask: U-Net predicted mask.

    Overlay legend:
    - Red (FN): Positive pixels incorrectly predicted as negative.
    - Green (TP): Positive pixels correctly predicted as positive.
    - Blue (FP): Negative pixels incorrectly predicted as positive;
    """
    overlay = np.zeros((*loose_mask.shape, 3), dtype=np.uint8)
    overlay[(loose_mask == 1) & (pred_mask == 0)] = [255, 90, 95]  # Red for FN
    overlay[(loose_mask == 1) & (pred_mask == 1)] = [77, 170, 87]  # Green for TP
    overlay[(loose_mask == 0) & (pred_mask == 1)] = [91, 192, 235]  # Blue for FP

    fig, axes = plt.subplots(1, 2, figsize=(80, 40), dpi=100)
    for ax in axes:
        ax.axis('off')

    axes[0].imshow(input_image, cmap='gray')
    axes[1].imshow(overlay)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
