import numpy as np
import random
import torch
import cv2


class MinMaxNormalize:
    """
    Normalizes the image using min-max normalization.
    """

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        min_val, max_val = torch.min(image), torch.max(image)
        return (image - min_val) / (max_val - min_val) if max_val > min_val else image


class LogTransform:
    """
    Applies logarithmic transformation to the image.
    """

    def __call__(self, image: torch.Tensor):
        return torch.log1p(image)


class PercentileNormalize:
    """
    Normalizes the image by dividing by the 99th percentile value of the non-zero pixels.
    """

    def __init__(self, percentile=99):
        self.percentile = percentile

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image_np = image.numpy()
        perc_value = np.percentile(image_np[image_np > 0], self.percentile)  # Exclude zero values
        normalized_image = image_np / perc_value
        return torch.tensor(normalized_image, dtype=image.dtype)


class ClipIntensity:
    """
    Clips the intensity values of the image to a maximum threshold.
    """

    def __init__(self, min_threshold=4):
        self.min_threshold = min_threshold

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image_np = image.numpy()
        perc_99 = np.percentile(image_np[image_np > 0], 99)  # Exclude zero values
        final_threshold = max(perc_99, self.min_threshold)
        clipped_image_np = np.clip(image_np, None, final_threshold)
        return torch.tensor(clipped_image_np, dtype=image.dtype)


class GaussianBlur:
    """
    Applies Gaussian blur to the image.
    """

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, image: torch.Tensor):
        image_np = image.squeeze(0).numpy()
        blurred_image_np = cv2.GaussianBlur(image_np, (self.kernel_size, self.kernel_size), 0)
        blurred_image = torch.tensor(blurred_image_np, dtype=image.dtype).unsqueeze(0)
        return blurred_image


class HistogramEqualization:
    """
    Applies histogram equalization to the image.
    """

    def __call__(self, image: torch.Tensor):
        image_np = image.squeeze(0).numpy().astype(np.uint8)
        equalized_image_np = cv2.equalizeHist(image_np)
        equalized_image = torch.tensor(equalized_image_np, dtype=image.dtype).unsqueeze(0)
        return equalized_image


class Random90Rotation:
    """
    Randomly rotates the image and mask by 0, 90, 180, or 270 degrees.
    """

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        k = random.randint(0, 3)  # Rotate 0, 90, 180, or 270 degrees
        return torch.rot90(image, k=k, dims=[1, 2]), torch.rot90(mask, k=k, dims=[1, 2])


class RandomFlip:
    """
    Randomly flips the image and mask horizontally and/or vertically.
    """

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() > 0.5:
            image, mask = torch.flip(image, dims=[2]), torch.flip(mask, dims=[2])  # Horizontal flip
        if random.random() > 0.5:
            image, mask = torch.flip(image, dims=[1]), torch.flip(mask, dims=[1])  # Vertical flip
        return image, mask


class ComposeTransformations:
    """
    Applies multiple transformations in a defined order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        for transform in self.transforms:
            if isinstance(transform, (Random90Rotation, RandomFlip)):
                image, mask = transform(image, mask)  # Apply augmentations to both
            else:
                image = transform(image)  # Apply transformations to image only
        return image, mask
