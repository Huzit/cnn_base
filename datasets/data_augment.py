import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform(
    img_size = (512, 512),
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
):

    """
    학습 데이터 전처리 및 증강
    :param img_size: 모델 입력 이미지 사이즈
    :param mean: 정규화할 RGB 평균
    :param std: 정규화할 표준편차
    :return:
    """
    height, width = img_size
    return A.Compose([
        #이미지 크기와 마스크 크기 통일
        A.Resize(height=height, width=width),
        #기본적인 공간 증강
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(
            limit=15,
            border_mode=0,
            p=0.3
        ),
        #이미지 색상 정규화
        A.Normalize(mean=mean, std=std),
        #Numpy 배열을 pyTorch Tensor로 변환
        ToTensorV2(),
    ])

def get_valid_transform(
        image_size=(512, 512),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
):
    """
    검증 및 테스트 데이터 전처리
    """
    height, width = image_size

    return A.Compose([
        A.Resize(
            height=height,
            width=width,
        ),
        A.Normalize(
            mean=mean,
            std=std,
        ),
        ToTensorV2(),
    ])