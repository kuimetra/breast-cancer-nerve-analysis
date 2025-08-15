import transforms as tfms

from typing import Dict
import tifffile as tiff
import numpy as np
import yaml
import os


def load_tiff_files(directory_path: str) -> dict[str, np.ndarray]:
    """
    Reads all TIFF files from the given directory and loads them into a dictionary.

    Parameters:
    - directory_path: Path to the directory containing the TIFF files.

    Returns:
    - images_dict: Dictionary where the key is the filename, and the value is the loaded image.
    """
    images_dict = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.tiff'):
            file_path = os.path.join(directory_path, filename)
            try:
                image = tiff.imread(file_path)

                if not np.issubdtype(image.dtype, np.floating):
                    image = image.astype(np.float32)

                images_dict[os.path.splitext(filename)[0]] = image
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return images_dict


def save_tiff_file(directory_path: str, filename: str, image: np.ndarray, binary: bool = False):
    """
    Saves a numpy array as a TIFF file in the specified directory.

    Parameters:
    - directory_path: Path to the directory where the TIFF file will be saved.
    - filename: Name of the TIFF file.
    - image: Image data to be saved.
    - binary: If True, the image is converted to binary before saving.
    """
    os.makedirs(directory_path, exist_ok=True)
    if binary:
        image = (image > 0).astype(np.uint8)
    tiff.imwrite(os.path.join(directory_path, f"{filename}.tiff"), image)


def load_config(config_path: str):
    """
    Loads the training configuration from a YAML file.

    Parameters:
    - config_path: Path to the YAML configuration file.

    Returns:
    - List of transformation objects based on the configuration.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    transformation_mapping = {
        "MinMaxNormalize": tfms.MinMaxNormalize,
        "LogTransform": tfms.LogTransform,
        "PercentileNormalize": tfms.PercentileNormalize,
        "ClipIntensity": tfms.ClipIntensity,
        "GaussianBlur": tfms.GaussianBlur,
        "HistogramEqualization": tfms.HistogramEqualization,
    }

    transformations = [transformation_mapping[name]() for name in config["transformations"] if
                       name in transformation_mapping]

    return transformations


def save_config(config: Dict, output_folder: str):
    """
    Saves the training configuration to a YAML file.

    Parameters:
    - config: Dictionary containing training configuration parameters.
    - output_folder: Directory where the configuration file will be saved.
    """
    config_path = os.path.join(output_folder, "config.yaml")

    config_serializable = {
        "epochs": config["num_epochs"],
        "patience": config["patience"],
        "early_stopping_triggered": config.get("early_stopping_epoch", "no"),
        "learning_rate": config["learning_rate"],
        "loss_function": config["loss_function"].name,
        "data_augmentation": config["data_augmentation"],
        "transformations": [t.__class__.__name__ for t in config["transformations"]]
    }

    with open(config_path, "w") as f:
        yaml.dump(config_serializable, f, default_flow_style=False)

    print(f"Training config saved at: {config_path}")


def get_file_paths(base_path: str = "data/datasets", subset: str = "train") -> Dict[str, str]:
    """
    Gets the file paths for the specified subset (train, val, or test).

    Parameters:
    - base_path: Base path to the dataset directory.
    - subset: Subset type ('train', 'val', 'test').

    Returns:
    - Dictionary containing paths for the 'peripherin_marker' and 'bin_nerve_mask'.
    """
    subset_path = os.path.join(base_path, subset)
    return {
        "peripherin_marker": os.path.join(subset_path, "peripherin_marker"),
        "bin_nerve_mask": os.path.join(subset_path, "bin_nerve_mask"),
    }


def load_data(base_path: str = "data/datasets", mode: str = "train") -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads TIFF files for the specified mode.

    Parameters:
    - base_path: Base path to the dataset directory.
    - mode: Mode to load data for ('train', 'val', 'test', or 'all').

    Returns:
    - Dictionary containing the loaded TIFF files for the specified mode.
    """
    if mode == "train":
        subsets = ["train", "val"]
    elif mode == "val":
        subsets = ["val"]
    elif mode == "test":
        subsets = ["test"]
    elif mode == "all":
        subsets = ["train", "val", "test"]
    else:
        raise ValueError("Invalid mode. Choose from 'train', 'val', 'test', or 'all'.")

    return {
        subset: {
            key: load_tiff_files(path)
            for key, path in get_file_paths(base_path, subset).items()
        }
        for subset in subsets
    }
