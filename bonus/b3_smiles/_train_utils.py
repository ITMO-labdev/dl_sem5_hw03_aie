"""Детерминизм и оценка MSE для ESOL-регрессоров.

Выдаётся заполненным как utility. **Не изменяйте этот файл.**

Функции:
    set_seed     — фиксирует все источники случайности.
    evaluate_mse — считает MSE модели на DataLoader'е без градиентов.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Фиксирует seed'ы всех генераторов случайности.

    Устанавливает seed для ``random`` (стандартная библиотека),
    ``numpy``, ``torch`` (CPU) и ``torch.cuda`` (все GPU, если доступны).

    Args:
        seed: Неотрицательное целое значение seed.

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
def evaluate_mse(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> float:
    """Вычисляет среднеквадратичную ошибку модели на всём ``loader``.

    Паттерн:
        1. Сохранить исходный режим модели (``was_training``).
        2. Перевести в ``eval()`` и прогнать все батчи без градиентов.
        3. Накопить сумму квадратов отклонений и число примеров
           (усреднение по примерам, а не по батчам — это важно, если
           последний батч неполный).
        4. Восстановить исходный режим модели в ``finally``.

    При пустом ``loader`` возвращает ``0.0``.

    Args:
        model: Модель с сигнатурой
            ``forward(input_ids, attention_mask) -> (B, 1)``.
        loader: DataLoader, отдающий словари с ключами ``input_ids``,
            ``attention_mask``, ``target``.
        device: Устройство для вычислений.

    Returns:
        MSE по всем примерам в ``loader`` (в тех же единицах, в которых
        заданы таргеты — то есть в нормализованных z-единицах).
    """
    was_training = model.training
    model.eval()

    total_sq = 0.0
    total_n = 0
    try:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            preds = model(input_ids, mask).squeeze(-1)
            total_sq += ((preds - target) ** 2).sum().item()
            total_n += target.numel()
    finally:
        if was_training:
            model.train()

    return total_sq / max(total_n, 1)