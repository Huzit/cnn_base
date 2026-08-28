import torch

from tqdm.auto import tqdm


def calculate_metrics(
        logits,
        targets,
        threshold=0.5,
        smooth=1e-6,
):
    """
    Binary Segmentation의 Dice와 IoU 계산
    """

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()

    predictions = predictions.view(predictions.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (predictions * targets).sum(dim=1)

    prediction_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)

    dice = (
                   2.0 * intersection + smooth
           ) / (
                   prediction_sum + target_sum + smooth
           )

    union = prediction_sum + target_sum - intersection

    iou = (
                  intersection + smooth
          ) / (
                  union + smooth
          )

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
    }