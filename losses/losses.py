import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    예측 마스크와 실제 마스크의 겹침 정도를 계산하는 Dice Loss
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Logits를 0~1 확률값으로 변환
        probabilities = torch.sigmoid(logits)

        # Batch별로 펼치기
        probabilities = probabilities.view(probabilities.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probabilities * targets).sum(dim=1)

        dice_score = (
                             2.0 * intersection + self.smooth
                     ) / (
                             probabilities.sum(dim=1)
                             + targets.sum(dim=1)
                             + self.smooth
                     )

        return 1.0 - dice_score.mean()


class BCEDiceLoss(nn.Module):
    """
    BCE Loss와 Dice Loss를 결합한 Binary Segmentation Loss
    """

    def __init__(
            self,
            bce_weight=0.5,
            dice_weight=0.5,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total_loss = (
                self.bce_weight * bce
                + self.dice_weight * dice
        )

        return total_loss