from pathlib import Path

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class SegDataset(Dataset):
    """
    이미지와 JSON Annotation을 사용하는
    Binary Segmentation Dataset 보일러플레이트
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(row["img_path"])
        json_path = row.get("json_path", None)

        # 이미지 불러오기
        image = cv2.imread(str(img_path))

        if image is None:
            raise RuntimeError(f"이미지 로드 실패: {img_path}")

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # 기본 빈 마스크 생성
        mask = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        # JSON Annotation이 있으면 마스크 생성
        if (
                isinstance(json_path, str)
                and Path(json_path).exists()
        ):
            mask, _ = seg_utils.json_to_mask(
                Path(json_path),
                img_path,
            )

        # Binary Mask 변환
        mask = (mask > 0).astype(np.uint8)

        # 전처리 및 데이터 증강
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

            mask = mask.float()

        else:
            # Image: HWC → CHW
            image = torch.from_numpy(
                image.transpose(2, 0, 1)
            ).float() / 255.0

            # Mask: HW → CHW
            mask = torch.from_numpy(
                mask
            ).unsqueeze(0).float()

        return {
            "image": image,
            "mask": mask,
            "class_name": int(row["class_name"]),
            "img_path": str(img_path),
        }