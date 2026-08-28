import torch


def build_optimizer(
        model,
        optimizer_name="adamw",
        learning_rate=1e-4,
        weight_decay=1e-5,
):
    """
    설정에 따라 Optimizer 생성
    """

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"지원하지 않는 Optimizer입니다: {optimizer_name}"
    )