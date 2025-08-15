import torch.nn as nn
import kornia
import torch


class BaseLoss(nn.Module):
    @property
    def name(self):
        return self.__class__.__name__


class WeightedCrossEntropyLoss(BaseLoss):
    def __init__(self, epsilon: float = 1e-5):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Parameters:
        - preds: Predicted logits, shape: (batch, 1, H, W)
        - targets: Ground truth binary mask, shape: (batch, 1, H, W)
        """
        probs = torch.sigmoid(preds)

        N = torch.numel(probs)

        sum_p = probs.sum()  # Sum of predicted foreground probabilities
        alpha = (N - sum_p) / (sum_p + self.epsilon)

        loss = - (1 / N) * torch.sum(alpha * targets * torch.log(probs + self.epsilon) +
                                     (1 - targets) * torch.log(1 - probs + self.epsilon))
        return loss


class FocalLoss(BaseLoss):
    def __init__(self, alpha: float = 0.99, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Parameters:
        - alpha: Weighting factor for the foreground class.
        - gamma: Focusing parameter that controls the rate at which easy examples are down-weighted.
        - reduction: The reduction applied to the output.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss = kornia.losses.binary_focal_loss_with_logits

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Parameters:
        - preds: Predicted logits, shape: (batch, 1, H, W)
        - targets: Ground truth binary mask, shape: (batch, 1, H, W)
        """
        loss = self.loss(preds, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        return loss


class DiceLoss(BaseLoss):
    def __init__(self, epsilon: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Parameters:
        - preds: Predicted logits, shape: (batch, 1, H, W)
        - targets: Ground truth binary mask, shape: (batch, 1, H, W)
        """
        probs = torch.sigmoid(preds)

        intersection = 2 * torch.sum(probs * targets)
        denominator = torch.sum(probs) + torch.sum(targets)

        dice_score = (intersection + self.epsilon) / (denominator + self.epsilon)
        return 1 - dice_score
