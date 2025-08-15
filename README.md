# Neural Structures in Breast Cancer: Segmentation and Survival Outcomes

This repository contains scripts developed for the Master's thesis *Neural Structures in Breast Cancer: Segmentation and
Survival Outcomes*. The project combines deep learning-based image segmentation with statistical survival analysis to
investigate nerve fiber elements in tumor tissue examined by Imaging Mass Cytometry (IMC).

---

## Repository Structure

### `pre_post_processing_scripts/`

- **`anonymize_tma.ipynb`** – Anonymizes TMA case IDs by assigning random identifiers.
- **`bin_to_obj_num.ipynb`** – Applies connected components labeling to binary masks (Subsection 3.3.4).
- **`cell_masks_preprocessing.ipynb`** – Converts single-layer cell masks into 10 layers, each representing a specific
  cell type.
- **`data_prep_for_survival_analysis.ipynb`** – Merges spatial distribution measurements with patient metadata and
  combines train/validation cases for survival analysis.
- **`mask_filtering.ipynb`** – Removes small objects from segmentation masks to reduce false positives (Subsection
  3.3.4).
- **`spatial_distribution.ipynb`** – Quantifies nerve overlap with tumor/stroma compartments and calculates cell-type
  composition within nerve regions (Section 3.4).

---

### `survival_analysis/`

- **`ph_assumption_example.Rmd`** – Generates Schoenfeld residual plots to demonstrate proportional hazards assumption
  checks with and without violations (Figure 2.23).
- **`survival_analysis.Rmd`** – Performs Cox regression analysis, including model fitting, evaluation, Schoenfeld plots,
  and statistical tests for assumption checking.

---

### Root scripts and notebooks

- **`data_analysis.ipynb`** – Exploratory data analysis (Section 4.1).
- **`data_io.py`** – Functions for loading/saving TIFF images, reading/writing YAML configs, and loading IMC data
  subsets for training/validation/testing.
- **`data_split.ipynb`** – Splits cases into train, validation, and test sets with nerve-count stratification;
  visualizes nerve count distributions.
- **`dataloader_utils.py`** – Builds PyTorch datasets and dataloaders for IMC images and nerve masks.
- **`figures.ipynb`** – Generates thesis figures, including preprocessing comparisons and overlays of loose ground truth
  vs. predicted nerve masks (Figures 2.3, 2.4, 2.5, 4.11, 4.12).
- **`losses.py`** – Defines segmentation losses: Weighted Cross-Entropy, Focal Loss, and Dice Loss.
- **`metrics_utils.py`** – Functions for model inference (with/without test-time augmentation) and evaluation.
- **`model.py`** – U-Net segmentation model architecture.
- **`model_selection.ipynb`** – Implements the model selection procedure described in Subsection 3.3.3.
- **`tma_case_lookup.py`** – Retrieves original TMA case IDs from anonymized IDs using a mapping file.
- **`train_grid.py`** – Performs grid search over preprocessing transformations, loss functions, and learning rates (
  Subsection 3.3.2).
- **`transforms.py`** – Preprocessing and augmentation transformations (normalization, intensity clipping, Gaussian
  blurring, histogram equalization, random rotations/flips).
- **`visualization.py`** – Plotting utilities for IMC images, cell masks, training loss curves, and prediction
  overlays (Figures 4.1, 4.3, 4.13, 4.16, A.1, A.2, A.3).

---

*Note: Due to the sensitivity of patient information and ethical restrictions, all datasets used in this project are excluded from this repository. Access may be granted upon request to the study administrator.*
