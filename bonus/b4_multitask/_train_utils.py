"""Утилиты для обучения: сидирование RNG и multi-task evaluation.

Выдаётся заполненным. **Не изменяйте этот файл.**

Функции:
    set_seed: детерминированное сидирование Python / NumPy / PyTorch.
    evaluate_multitask: оценка multi-task модели (accuracy + MSE) на loader'е.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Фиксирует RNG Python / NumPy / PyTorch для воспроизводимости.

    Args:
        seed: Неотрицательное целое.

    Raises:
        ValueError: Если ``seed < 0``.
    """
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_multitask(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> dict[str, float]:
    """Считает classification accuracy и regression MSE на loader'е.

    Переключает модель в ``eval()``-режим, а после оценки восстанавливает
    исходный режим (``train()`` или ``eval()``) — это важно при вызове
    из тренировочного цикла, иначе следующая эпоха пойдёт с выключенным
    dropout и batchnorm'ом в eval-режиме.

    Args:
        model: Multi-task модель, ``forward(x)`` возвращает словарь с
            ключами ``"cls_logits"`` и ``"reg_pred"``.
        loader: DataLoader, выдающий словари с ключами ``"image"``,
            ``"cls_target"``, ``"reg_target"``.
        device: Устройство для инференса.

    Returns:
        Словарь с ключами:
            - ``"cls_accuracy"``: доля правильных предсказаний класса,
            - ``"reg_mse"``: средний MSE по регрессионной голове.
    """
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    sq_err = 0.0

    try:
        for batch in loader:
            images = batch["image"].to(device)
            cls_t = batch["cls_target"].to(device)
            reg_t = batch["reg_target"].to(device)

            out = model(images)
            preds = out["cls_logits"].argmax(dim=-1)
            correct += (preds == cls_t).sum().item()
            total += cls_t.numel()

            reg_pred = out["reg_pred"].squeeze(-1)
            sq_err += ((reg_pred - reg_t) ** 2).sum().item()
    finally:
        if was_training:
            model.train()

    return {
        "cls_accuracy": correct / max(total, 1),
        "reg_mse": sq_err / max(total, 1),
    }