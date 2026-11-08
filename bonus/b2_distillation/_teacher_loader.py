"""Загрузчик предобученного teacher-ViT из задания 4.2.

Выдаётся заполненным как utility. **Не изменяйте этот файл.**

Функции:
    load_teacher — загружает fine-tuned ViT-Tiny из чекпоинта
                   ``results/model.pt`` и замораживает все параметры.
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn


def load_teacher(
    checkpoint_path: str | Path = "results/model.pt",
    num_classes: int = 10,
    device: str = "cpu",
) -> nn.Module:
    """Загружает fine-tuned ViT-Tiny teacher'а, обученного в задании 4.2.

    Алгоритм:
        1. Создать архитектуру ``vit_tiny_patch16_224`` через
           ``timm.create_model`` **без предобученных весов**
           (``pretrained=False``) — они будут перезаписаны из чекпоинта.
        2. Заменить классификационную голову на ``nn.Linear`` под
           ``num_classes`` (архитектура должна точно совпадать с той,
           что использовалась при сохранении чекпоинта в задании 4.2).
        3. Загрузить ``state_dict`` из ``checkpoint_path`` с
           ``weights_only=True`` (безопасная загрузка — без выполнения
           произвольного кода).
        4. Переместить модель на ``device``, перевести в ``eval()``
           и заморозить все параметры (``requires_grad = False``).

    Args:
        checkpoint_path: Путь к ``model.pt``, сохранённому скриптом
            ``scripts.train_vit`` (задание 4.2).
        num_classes: Число классов. Должно совпадать с чекпоинтом
            (10 для EuroSAT).
        device: Устройство, на которое загружается модель.

    Returns:
        Teacher-модель в режиме ``eval()`` с ``requires_grad=False``
        на всех параметрах.

    Raises:
        FileNotFoundError: Если файл ``checkpoint_path`` не найден.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {checkpoint_path}. "
            "Run `python -m scripts.train_vit` first (task 4.2)."
        )

    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)

    state_dict = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model