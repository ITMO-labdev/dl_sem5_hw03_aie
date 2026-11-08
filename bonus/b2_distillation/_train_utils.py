"""Вспомогательные утилиты для обучения с дистилляцией.

Выдаётся заполненным как utility. **Не изменяйте этот файл.**

Функции:
    set_seed — фиксирует все источники случайности для воспроизводимости.
    evaluate — вычисляет среднюю loss и accuracy на DataLoader'е
               без вычисления градиентов.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Фиксирует seed'ы всех генераторов случайности.

    Устанавливает seed для:
        - ``random`` (стандартная библиотека Python),
        - ``numpy``,
        - ``torch`` (CPU),
        - ``torch.cuda`` (все GPU, если доступны).

    Args:
        seed: Значение seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Вычисляет среднюю loss и accuracy модели на ``loader``.

    Паттерн:
        1. Сохранить исходный режим модели (``was_training``).
        2. Перевести в ``eval()``, прогнать все батчи в
           ``torch.inference_mode()`` (быстрее ``no_grad`` — отключает
           также view-tracking).
        3. Накопить взвешенные (по размеру батча) сумму loss
           и число правильных предсказаний.
        4. Восстановить исходный режим модели в ``finally``.

    При пустом ``loader`` возвращает ``(0.0, 0.0)``.

    Args:
        model: Модель для оценки.
        loader: DataLoader, отдающий ``(images, labels)``.
        criterion: Loss-функция (например, ``nn.CrossEntropyLoss()``).
        device: Устройство для вычислений.

    Returns:
        Кортеж ``(average_loss, accuracy)``, где:
            - ``average_loss``: средняя loss, взвешенная по размеру батча;
            - ``accuracy``: доля правильных предсказаний в ``[0, 1]``.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    try:
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += batch_size
    finally:
        if was_training:
            model.train()

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples