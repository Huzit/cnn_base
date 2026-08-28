import segmentation_models_pytorch as smp


def build_model(
        architecture="unet",
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
):
    """
    Segmentation 모델 생성

    Args:
        architecture:
            모델 구조
            unet, unetplusplus, deeplabv3plus, fpn

        encoder_name:
            Encoder Backbone
            resnet34, efficientnet-b0 등

        encoder_weights:
            사전학습 가중치
            imagenet 또는 None

        in_channels:
            입력 이미지 채널 수

        num_classes:
            출력 클래스 수
            Binary Segmentation은 1
    """

    model_list = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
    }

    architecture = architecture.lower()

    if architecture not in model_list:
        raise ValueError(
            f"지원하지 않는 모델입니다: {architecture}\n"
            f"사용 가능: {list(model_list.keys())}"
        )

    model = model_list[architecture](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )

    return model