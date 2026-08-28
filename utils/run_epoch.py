from cv2.datasets import none
from jupyter_lsp import non_blocking
from torch.optim import Optimizer
import matrix as m


def run_epoch(
    model,
    data_loader,
    criterion,
    device,
    optimizer: Optimizer=None,
    threshold=0.5
):
    """
    기본적으로 영상 학습을 기반으로 설계된 함수입니다.
    기본적으로 Dice_score와 Iou_score를 계산합니다.
    사용자의 요구에 따라 별도의 평가 지표를 위한 matrix를 추가하셔도 무방합니다.

    :param model: 백본 입력
    :param data_loader: test/val에 따라 다르게 학습
    :param criterion: 목적 함수
    :param device: cuda device
    :param optimizer: 최적화 함수
    :param threshold: 임계값
    :return: loss, dice, iou 값
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
        mode = "Train"
    else:
        model.eval()
        mode = "Val"

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    progress_bar = tqdm(data_loader, desc=mode, leave=False)

    for batch in progress_bar:
        images = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        # Train일때만 Gradient 초기화
        if is_training:
            optimizer.zero_grad()

        #Train에서는 Gradient 계산
        #Val / Test에서는 Gradient 계산하지 않음
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, mask)

            if is_training:
                loss.backward()
                optimizer.step()


        metrics = m.calculate_metrics(
            logits.detach(),
            target = mask,
            threshold=threshold
        )

        batch_size = image.size(0)

        total_loss += loss.item() * batch_size
        total_dice += metrics["dice"] * batch_size
        total_iou += metrics["iou"] * batch_size

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{metrics['dice']:.4f}",
            "iou": f"{metrics['iou']:.4f}",
        })

    dataset_size = len(data_loader.dataset)

    return {
        "loss": total_loss / dataset_size,
        "dice": total_dice / dataset_size,
        "iou": total_iou / dataset_size,
    }
