from pathlib import Path

import cv2
import numpy as np
import torch


def load_checkpoint(
        model,
        checkpoint_path,
        device,
):
    """
    저장된 Checkpoint에서 모델 가중치 불러오기
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model


@torch.no_grad()
def predict_image(
        model,
        image_path,
        transform,
        device,
        threshold=0.5,
):
    """
    단일 이미지에 대해 Segmentation 추론 수행
    """

    image_path = Path(image_path)

    # 이미지 로드
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"이미지 로드 실패: {image_path}"
        )

    # BGR → RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    original_height, original_width = image.shape[:2]

    # 전처리
    transformed = transform(image=image)
    input_tensor = transformed["image"]

    # (C, H, W) → (1, C, H, W)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 모델 추론
    logits = model(input_tensor)

    # Logits → Probability
    probability = torch.sigmoid(logits)

    # Probability → Binary Mask
    prediction = (
            probability >= threshold
    ).float()

    # (1, 1, H, W) → (H, W)
    probability = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # 원본 이미지 크기로 복원
    probability = cv2.resize(
        probability,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    prediction = cv2.resize(
        prediction,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    return {
        "image": image,
        "probability": probability,
        "mask": prediction,
        "image_path": str(image_path),
    }